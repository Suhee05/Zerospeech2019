import os
import sys
import numpy as np
import tensorflow as tf
from utils.audio import save_wavenet_wav, get_hop_size, melspectrogram
from utils.infolog import log
from train import create_model
from feeder import _interp

from utils import util


class Synthesizer:
	def load(self, checkpoint_path, hparams, model_name='WaveNet'):
		log('Constructing model: {}'.format(model_name))
		self._hparams = hparams
		local_cond, global_cond = self._check_conditions()

		self.local_conditions = tf.placeholder(tf.float32, shape=(None, None, hparams.num_mfccs), name='local_condition_features') if local_cond else None
		self.global_conditions = tf.placeholder(tf.int32, shape=(None, 1), name='global_condition_features') if global_cond else None
		self.synthesis_length = tf.placeholder(tf.int32, shape=(), name='synthesis_length') if not local_cond else None
		self.input_lengths = tf.placeholder(tf.int32, shape=(1, ), name='input_lengths') if hparams.wavenet_synth_debug else None
		self.synth_debug = hparams.wavenet_synth_debug

		with tf.variable_scope('WaveNet_model') as scope:
			self.model = create_model(model_name, hparams)
			self.model.initialize(y=None, c=self.local_conditions, g=self.global_conditions,
				input_lengths=self.input_lengths, synthesis_length=self.synthesis_length, test_inputs=None)

			self._hparams = hparams
			sh_saver = create_shadow_saver(self.model)

			log('Loading checkpoint: {}'.format(checkpoint_path))
			#Memory allocation on the GPU as needed
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			config.allow_soft_placement = True

			self.session = tf.Session(config=config)
			self.session.run(tf.global_variables_initializer())

		load_averaged_model(self.session, sh_saver, checkpoint_path)


	def synthesize(self, mel_spectrograms, speaker_ids, basenames, out_dir, log_dir, embed_dir, embed_only=True):
		hparams = self._hparams
		local_cond, global_cond = self._check_conditions()

		#Switch mels in case of debug
		if self.synth_debug:
			assert len(hparams.wavenet_debug_mels) == len(hparams.wavenet_debug_wavs)
			mel_spectrograms = [np.load(mel_file) for mel_file in hparams.wavenet_debug_mels]


		#Prepare local condition batch
		maxlen = max([len(x) for x in mel_spectrograms])
		#[-max, max] or [0,max]
		T2_output_range = (-self._hparams.max_abs_value, self._hparams.max_abs_value) if self._hparams.symmetric_mels else (0, self._hparams.max_abs_value)

		if self._hparams.clip_for_wavenet:
			mel_spectrograms = [np.clip(x, T2_output_range[0], T2_output_range[1]) for x in mel_spectrograms]


		c_batch = np.asarray(mel_spectrograms).astype(np.float32)
		print("c batch shape {}".format(c_batch.shape))
		if self._hparams.normalize_for_wavenet:
			#rerange to [0, 1]
			c_batch = _interp(c_batch, T2_output_range).astype(np.float32)

		g = None if speaker_ids is None else np.asarray(speaker_ids, dtype=np.int32).reshape(len(c_batch), 1)
		print("g shape {}".format(g.shape))
		feed_dict = {}

		if local_cond:
			feed_dict[self.local_conditions] = c_batch
		else:
			feed_dict[self.synthesis_length] = 100

		if global_cond:
			feed_dict[self.global_conditions] = g

		if self.synth_debug:
			debug_wavs = hparams.wavenet_debug_wavs
			assert len(debug_wavs) % hparams.wavenet_num_gpus == 0
			test_wavs = [np.load(debug_wav).reshape(-1, 1) for debug_wav in debug_wavs]

			#pad wavs to same length
			max_test_len = max([len(x) for x in test_wavs])
			test_wavs = np.stack([_pad_inputs(x, max_test_len) for x in test_wavs]).astype(np.float32)

			assert len(test_wavs) == len(debug_wavs)
			#### GTA False
			feed_dict[self.input_lengths] = np.asarray([test_wavs.shape[1]])

        if embed_only == False:

            #Generate wavs and clip extra padding to select Real speech parts
            #### VQVAE Out
            generated_wavs, upsampled_features, vq_embeddings, vq_onehot,vq_w,vq_enc_ind = self.session.run([self.model.tower_y_hat, self.model.tower_synth_upsampled_local_features, self.model.vq_embeddings, self.model.vq_onehot,self.model.vq_w,self.model.vq_enc_ind], feed_dict=feed_dict)

            #Linearize outputs (n_gpus -> 1D)
            generated_wavs = [wav for gpu_wavs in generated_wavs for wav in gpu_wavs]
            upsampled_features = [feat for gpu_feats in upsampled_features for feat in gpu_feats]


            for i, (generated_wav, input_mel, upsampled_feature, vq_embedding) in enumerate(zip(generated_wavs, mel_spectrograms, upsampled_features, vq_embeddings)):
                #Save wav to disk
                audio_filename = os.path.join(out_dir, 'wavenet-audio-{}.wav'.format(basenames[i]))
                save_wavenet_wav(generated_wav, audio_filename, sr=hparams.sample_rate, inv_preemphasize=hparams.preemphasize, k=hparams.preemphasis)

                #### Vq embedding save (shape [batch_size, num_frames, embed_dim])
                embed_filename = os.path.join(embed_dir, 'emb-{}.npy'.format(basenames[i]))
                np.save(embed_filename, vq_embedding)

                onehot_filename = os.path.join(embed_dir, 'onehot-{}.npy'.format(basenames[i]))
                np.save(onehot_filename, vq_onehot)

                wmatrix_filename = os.path.join(embed_dir, 'wmatrix-{}.npy'.format(basenames[i]))
                np.save(wmatrix_filename, vq_w)

                idx_filename = os.path.join(embed_dir, 'idx-{}.npy'.format(basenames[i]))
                np.save(idx_filename, vq_enc_ind)


                #Compare generated wav mel with original input mel to evaluate wavenet audio reconstruction performance
                #Both mels should match on low frequency information, wavenet mel should contain more high frequency detail when compared to Tacotron mels.
                generated_mel = melspectrogram(generated_wav, hparams).T
                util.plot_spectrogram(generated_mel, os.path.join(log_dir, 'wavenet-mel-spectrogram-{}.png'.format(basenames[i])),
                        title='Local Condition vs Reconstructed Audio Mel-Spectrogram analysis', target_spectrogram=input_mel)
                #Save upsampled features to visualize checkerboard artifacts.
                util.plot_spectrogram(upsampled_feature.T, os.path.join(log_dir, 'wavenet-upsampled_features-{}.png'.format(basenames[i])),
                        title='Upmsampled Local Condition features', auto_aspect=True)

                #Save waveplot to disk
                if log_dir is not None:
                    plot_filename = os.path.join(log_dir, 'wavenet-waveplot-{}.png'.format(basenames[i]))
                    util.waveplot(plot_filename, generated_wav, None, hparams, title='WaveNet generated Waveform.')


        else:
            #Generate wavs and clip extra padding to select Real speech parts
            #### VQVAE Out

            vq_embeddings,vq_onehot,vq_w,vq_enc_ind = self.session.run([self.model.vq_embeddings,self.model.vq_onehot,self.model.vq_w,self.model.vq_enc_ind], feed_dict=feed_dict)

            for i, vq_embedding in enumerate(vq_embeddings):

                #### Vq embedding save (shape [batch_size, num_frames, embed_dim])
                embed_filename = os.path.join(embed_dir, 'emb-{}.npy'.format(basenames[i]))
                np.save(embed_filename, vq_embedding)

                onehot_filename = os.path.join(embed_dir, 'onehot-{}.npy'.format(basenames[i]))
                np.save(onehot_filename, vq_onehot)

                wmatrix_filename = os.path.join(embed_dir, 'wmatrix-{}.npy'.format(basenames[i]))
                np.save(wmatrix_filename, vq_w)

                idx_filename = os.path.join(embed_dir, 'idx-{}.npy'.format(basenames[i]))
                np.save(idx_filename, vq_enc_ind)






	def _check_conditions(self):
		local_condition = self._hparams.cin_channels > 0
		global_condition = self._hparams.gin_channels > 0
		return local_condition, global_condition


def _pad_inputs(x, maxlen, _pad=0):
	return np.pad(x, [(0, maxlen - len(x)), (0, 0)], mode='constant', constant_values=_pad)

def create_shadow_saver(model, global_step=None):
	'''Load shadow variables of saved model.

	Inspired by: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

	Can also use: shadow_dict = model.ema.variables_to_restore()
	'''
	#Add global step to saved variables to save checkpoints correctly
	shadow_variables = [model.ema.average_name(v) for v in model.variables]
	variables = model.variables

	if global_step is not None:
		shadow_variables += ['global_step']
		variables += [global_step]

	shadow_dict = dict(zip(shadow_variables, variables))
	return tf.train.Saver(shadow_dict, max_to_keep=20)


def load_averaged_model(sess, sh_saver, checkpoint_path):
	sh_saver.restore(sess, checkpoint_path)




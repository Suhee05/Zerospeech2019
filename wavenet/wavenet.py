import numpy as np
import tensorflow as tf
from utils import audio
from utils.infolog import log
from utils import util
from utils.util import *
import vqvae.model as vqvae
import speaker_classifier.model as spkclassifier

from .gaussian import sample_from_gaussian
from .mixture import sample_from_discretized_mix_logistic
from .modules import (CausalConv1D, Conv1D1x1, ConvTranspose2D, ConvTranspose1D, ResizeConvolution, SubPixelConvolution, NearestNeighborUpsample, DiscretizedMixtureLogisticLoss,
	GaussianMaximumLikelihoodEstimation, MaskedMeanSquaredError, LeakyReluActivation, MaskedCrossEntropyLoss, ReluActivation, ResidualConv1DGLU, WeightNorm, Embedding)


def _expand_global_features(batch_size, time_length, global_features, data_format='BCT'):
	"""Expand global conditioning features to all time steps

	Args:
		batch_size: int
		time_length: int
		global_features: Tensor of shape [batch_size, channels] or [batch_size, channels, 1]
		data_format: string, 'BCT' to get output of shape [batch_size, channels, time_length]
			or 'BTC' to get output of shape [batch_size, time_length, channels]

	Returns:
		None or Tensor of shape [batch_size, channels, time_length] or [batch_size, time_length, channels]
	"""
	accepted_formats = ['BCT', 'BTC']
	if not (data_format in accepted_formats):
		raise ValueError('{} is an unknow data format, accepted formats are "BCT" and "BTC"'.format(data_format))

	if global_features is None:
		return None

	#[batch_size, channels] ==> [batch_size, channels, 1]
	# g = tf.cond(tf.equal(tf.rank(global_features), 2),
	# 	lambda: tf.expand_dims(global_features, axis=-1),
	# 	lambda: global_features)

	# g = tf.reshape(global_features, [tf.shape(global_features)[0], tf.shape(global_features)[1], 1])
	g = tf.reshape(global_features, [tf.shape(global_features)[0], 16, 1])
	g_shape = tf.shape(g)

	#[batch_size, channels, 1] ==> [batch_size, channels, time_length]
	# ones = tf.ones([g_shape[0], g_shape[1], time_length], tf.int32)
	# g = g * ones
	g = tf.tile(g, [1, 1, time_length])

	if data_format == 'BCT':
		return g

	else:
		#[batch_size, channels, time_length] ==> [batch_size, time_length, channels]
		return tf.transpose(g, [0, 2, 1])


def receptive_field_size(total_layers, num_cycles, kernel_size, dilation=lambda x: 2**x):
	"""Compute receptive field size.

	Args:
		total_layers; int
		num_cycles: int
		kernel_size: int
		dilation: callable, function used to compute dilation factor.
			use "lambda x: 1" to disable dilated convolutions.

	Returns:
		int: receptive field size in sample.
	"""
	assert total_layers % num_cycles == 0

	layers_per_cycle = total_layers // num_cycles
	dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
	return (kernel_size - 1) * sum(dilations) + 1

def maybe_Normalize_weights(layer, weight_normalization=True, init=False, init_scale=1.):
	"""Maybe Wraps layer with Weight Normalization wrapper.

	Args;
		layer: tf layers instance, the layer candidate for normalization
		weight_normalization: Boolean, determines whether to normalize the layer
		init: Boolean, determines if the current run is the data dependent initialization run
		init_scale: Float, Initialisation scale of the data dependent initialization. Usually 1.
	"""
	if weight_normalization:
		return WeightNorm(layer, init, init_scale)
	return layer

class WaveNet():
	"""Tacotron-2 Wavenet Vocoder model.
	"""
	def __init__(self, config, init):
		#Get config
		self._config = config

		# if self.local_conditioning_enabled():
		# 	assert config.num_mfccs == config.cin_channels

		#Initialize model architecture
		assert config.layers % config.stacks == 0
		layers_per_stack = config.layers // config.stacks

		self.scalar_input = is_scalar_input(config.input_type)

		#first (embedding) convolution
		with tf.variable_scope('input_convolution'):
			if self.scalar_input:
				self.first_conv = Conv1D1x1(config.residual_channels,
					weight_normalization=config.wavenet_weight_normalization,
					weight_normalization_init=init,
					weight_normalization_init_scale=config.wavenet_init_scale,
					name='input_convolution')
			else:
				self.first_conv = Conv1D1x1(config.residual_channels,
					weight_normalization=config.wavenet_weight_normalization,
					weight_normalization_init=init,
					weight_normalization_init_scale=config.wavenet_init_scale,
					name='input_convolution')

		#Residual Blocks
		self.residual_layers = []
		for layer in range(config.layers):
			self.residual_layers.append(ResidualConv1DGLU(
			config.residual_channels, config.gate_channels,
			kernel_size=config.kernel_size,
			skip_out_channels=config.skip_out_channels,
			use_bias=config.use_bias,
			dilation_rate=2**(layer % layers_per_stack),
			dropout=config.wavenet_dropout,
			cin_channels=config.cin_channels,
			gin_channels=config.gin_channels,
			weight_normalization=config.wavenet_weight_normalization,
			init=init,
			init_scale=config.wavenet_init_scale,
			residual_legacy=config.residual_legacy,
			name='ResidualConv1DGLU_{}'.format(layer)))

		#Final (skip) convolutions
		with tf.variable_scope('skip_convolutions'):
			self.last_conv_layers = [
			ReluActivation(name='final_conv_relu1'),
			# Conv1D1x1(config.skip_out_channels,
			# 	weight_normalization=config.wavenet_weight_normalization,
			# 	weight_normalization_init=init,
			# 	weight_normalization_init_scale=config.wavenet_init_scale,
			# 	name='final_convolution_1'),
			ReluActivation(name='final_conv_relu2'),]
			# Conv1D1x1(config.out_channels,
			# 	weight_normalization=config.wavenet_weight_normalization,
			# 	weight_normalization_init=init,
			# 	weight_normalization_init_scale=config.wavenet_init_scale,
			# 	name='final_convolution_2'),]


		#Global conditionning embedding
		if config.gin_channels > 0 and config.use_speaker_embedding:
			assert config.n_speakers is not None
			self.embed_speakers = Embedding(
				config.n_speakers, config.gin_channels, std=0.1, name='gc_embedding')
			self.embedding_table = self.embed_speakers.embedding_table
			# print("embedding table created : {}".format(self.embedding_table))
		else:
			self.embed_speakers = None

		self.all_convs = [self.first_conv] + self.residual_layers + self.last_conv_layers

		#Upsample conv net
		if self.local_conditioning_enabled():

			# VQVAE Model
			self.vqencoder = vqvae.VqEncoder(config)

			# Speaker Classifier model 											#spkclassifier
			self.spkclassifier = spkclassifier.SpeakerClassifier(config)		#spkclassifier

			#First convolutional layer for local condition
			with tf.variable_scope('lc_first_conv'):
				self.lc_first_conv = CausalConv1D(config.cin_first_conv_channels, config.cin_first_conv_kernel_size,
					weight_normalization=config.wavenet_weight_normalization,
					weight_normalization_init=init,
					weight_normalization_init_scale=config.wavenet_init_scale,
					name='lc_first_convolution')

			self.all_convs += [self.lc_first_conv]

			self.upsample_conv = []

			if config.upsample_type == 'NearestNeighbor':
				#Nearest neighbor upsampling (non-learnable)

				self.upsample_conv.append(NearestNeighborUpsample(strides=(1, audio.get_hop_size(config)*4)))

			else:
				#Learnable upsampling layers
				for i, s in enumerate(config.upsample_scales):
					with tf.variable_scope('local_conditioning_upsampling_{}'.format(i+1)):
						if config.upsample_type == '2D':
							convt = ConvTranspose2D(1, (config.freq_axis_kernel_size, s),
								padding='same', strides=(1, s), NN_init=config.NN_init, NN_scaler=config.NN_scaler,
								up_layers=len(config.upsample_scales), name='ConvTranspose2D_layer_{}'.format(i))

						elif config.upsample_type == '1D':
							convt = ConvTranspose1D(config.cin_channels, (s, ),
								padding='same', strides=(s, ), NN_init=config.NN_init, NN_scaler=config.NN_scaler,
								up_layers=len(config.upsample_scales), name='ConvTranspose1D_layer_{}'.format(i))

						elif config.upsample_type == 'Resize':
							convt = ResizeConvolution(1, (config.freq_axis_kernel_size, s),
								padding='same', strides=(1, s), NN_init=config.NN_init, NN_scaler=config.NN_scaler,
								up_layers=len(config.upsample_scales), name='ResizeConvolution_layer_{}'.format(i))

						else:
							assert config.upsample_type == 'SubPixel'
							convt = SubPixelConvolution(1, (config.freq_axis_kernel_size, 3),
								padding='same', strides=(1, s), NN_init=config.NN_init, NN_scaler=config.NN_scaler,
								up_layers=len(config.upsample_scales), name='SubPixelConvolution_layer_{}'.format(i))

						self.upsample_conv.append(maybe_Normalize_weights(convt,
							config.wavenet_weight_normalization, init, config.wavenet_init_scale))

						if config.upsample_activation == 'LeakyRelu':
							self.upsample_conv.append(LeakyReluActivation(alpha=config.leaky_alpha,
								name='upsample_leaky_relu_{}'.format(i+1)))
						elif config.upsample_activation == 'Relu':
							self.upsample_conv.append(ReluActivation(name='upsample_relu_{}'.format(i+1)))
						else:
							assert config.upsample_activation == None

			self.all_convs += self.upsample_conv

		self.receptive_field = receptive_field_size(config.layers,
			config.stacks, config.kernel_size)


	def set_mode(self, is_training):
		for conv in self.all_convs:
			try:
				conv.set_mode(is_training)
			except AttributeError:
				pass

	def initialize(self, y, c, g, input_lengths, x=None, synthesis_length=None, test_inputs=None, split_infos=None):
		'''Initialize wavenet graph for train, eval and test cases.
		'''
		config = self._config
		self.is_training = x is not None
		self.is_evaluating = not self.is_training and y is not None
		#Set all convolutions to corresponding mode
		self.set_mode(self.is_training)

		split_device = '/cpu:0' if self._config.wavenet_num_gpus > 1 or self._config.split_on_cpu else '/gpu:0'
		with tf.device(split_device):
			hp = self._config
			lout_int = [tf.int32] * hp.wavenet_num_gpus
			lout_float = [tf.float32] * hp.wavenet_num_gpus

			tower_input_lengths = tf.split(input_lengths, num_or_size_splits=hp.wavenet_num_gpus, axis=0) if input_lengths is not None else [input_lengths] * hp.wavenet_num_gpus

			tower_y = tf.split(y, num_or_size_splits=hp.wavenet_num_gpus, axis=0) if y is not None else [y] * hp.wavenet_num_gpus
			tower_x = tf.split(x, num_or_size_splits=hp.wavenet_num_gpus, axis=0) if x is not None else [x] * hp.wavenet_num_gpus
			tower_c = tf.split(c, num_or_size_splits=hp.wavenet_num_gpus, axis=0) if self.local_conditioning_enabled() else [None] * hp.wavenet_num_gpus
			tower_g = tf.split(g, num_or_size_splits=hp.wavenet_num_gpus, axis=0) if self.global_conditioning_enabled() else [None] * hp.wavenet_num_gpus
			tower_test_inputs = tf.split(test_inputs, num_or_size_splits=hp.wavenet_num_gpus, axis=0) if test_inputs is not None else [test_inputs] * hp.wavenet_num_gpus

		self.tower_y_hat_q = []
		self.tower_y_hat_train = []
		self.tower_y = []
		self.tower_input_lengths = []
		self.tower_means = []
		self.tower_log_scales = []
		self.tower_y_hat_log = []
		self.tower_y_log = []
		self.tower_c = []
		self.tower_y_eval = []
		self.tower_eval_length = []
		self.tower_y_hat = []
		self.tower_y_target = []
		self.tower_eval_c = []
		self.tower_mask = []
		self.tower_upsampled_local_features = []
		self.tower_eval_upsampled_local_features = []
		self.tower_synth_upsampled_local_features = []

		self.tower_spk_hat_train = []	#spkclassifier
		self.tower_g = []				#spkclassifier

		log('Initializing Wavenet model.  Dimensions (? = dynamic shape): ')
		log('  Train mode:                {}'.format(self.is_training))
		log('  Eval mode:                 {}'.format(self.is_evaluating))
		log('  Synthesis mode:            {}'.format(not (self.is_training or self.is_evaluating)))

		#1. Declare GPU devices
		gpus = ['/gpu:{}'.format(i) for i in range(hp.wavenet_num_gpus)]
		for i in range(hp.wavenet_num_gpus):
			with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device='/cpu:0', worker_device=gpus[i])):
				with tf.variable_scope('inference') as scope:
					log('  device:                    {}'.format(i))
					#Training
					if self.is_training:
						batch_size = tf.shape(x)[0]
						#[batch_size, time_length, 1]
						self.tower_mask.append(self.get_mask(tower_input_lengths[i], maxlen=tf.shape(tower_x[i])[-1])) #To be used in loss computation
						#[batch_size, channels, time_length]
						y_hat_train, spk_hat_train = self.step(tower_x[i], tower_c[i], tower_g[i], softmax=False) #softmax is automatically computed inside softmax_cross_entropy if needed  ##spkclassfier
						print("y_hat_train : {}".format(y_hat_train))
						print("spk_hat_train : {}".format(spk_hat_train))


						if is_mulaw_quantize(config.input_type):
							#[batch_size, time_length, channels]
							self.tower_y_hat_q.append(tf.transpose(y_hat_train, [0, 2, 1]))

						self.tower_y_hat_train.append(y_hat_train)
						self.tower_y.append(tower_y[i])
						self.tower_input_lengths.append(tower_input_lengths[i])


						self.tower_spk_hat_train.append(spk_hat_train)	#spkclassifier
						self.tower_g.append(tower_g[i])					#spkclassifier

						#Add mean and scale stats if using Guassian distribution output (there would be too many logistics if using MoL)
						if self._config.out_channels == 2:
							self.tower_means.append(y_hat_train[:, 0, :])
							self.tower_log_scales.append(y_hat_train[:, 1, :])
						else:
							self.tower_means.append(None)

						#Graph extension for log saving
						#[batch_size, time_length]
						shape_control = (batch_size, tf.shape(tower_x[i])[-1], 1)
						with tf.control_dependencies([tf.assert_equal(tf.shape(tower_y[i]), shape_control)]):
							y_log = tf.squeeze(tower_y[i], [-1])
							if is_mulaw_quantize(config.input_type):
								self.tower_y[i] = y_log

						y_hat_log = tf.cond(tf.equal(tf.rank(y_hat_train), 4),
							lambda: tf.squeeze(y_hat_train, [-1]),
							lambda: y_hat_train)
						y_hat_log = tf.reshape(y_hat_log, [batch_size, config.out_channels, -1])

						if is_mulaw_quantize(config.input_type):
							#[batch_size, time_length]
							y_hat_log = tf.argmax(tf.nn.softmax(y_hat_log, axis=1), 1)

							y_hat_log = util.inv_mulaw_quantize(y_hat_log, config.quantize_channels)
							y_log = util.inv_mulaw_quantize(y_log, config.quantize_channels)

						else:
							#[batch_size, time_length]
							if config.out_channels == 2:
								y_hat_log = sample_from_gaussian(
									y_hat_log, log_scale_min_gauss=config.log_scale_min_gauss)
							else:
								y_hat_log = sample_from_discretized_mix_logistic(
									y_hat_log, log_scale_min=config.log_scale_min)

							if is_mulaw(config.input_type):
								y_hat_log = util.inv_mulaw(y_hat_log, config.quantize_channels)
								y_log = util.inv_mulaw(y_log, config.quantize_channels)

						self.tower_y_hat_log.append(y_hat_log)
						self.tower_y_log.append(y_log)
						self.tower_c.append(tower_c[i])
						self.tower_upsampled_local_features.append(self.upsampled_local_features)

						log('  inputs:                    {}'.format(tower_x[i].shape))
						if self.local_conditioning_enabled():
							log('  local_condition:           {}'.format(tower_c[i].shape))
						if self.has_speaker_embedding():
							log('  global_condition:          {}'.format(tower_g[i].shape))
						log('  targets:                   {}'.format(y_log.shape))
						log('  outputs:                   {}'.format(y_hat_log.shape))


					#evaluating
					elif self.is_evaluating:
						#[time_length, ]
						idx = 0
						length = tower_input_lengths[i][idx]
						y_target = tf.reshape(tower_y[i][idx], [-1])[:length]
						test_inputs = tf.reshape(y_target, [1, -1, 1]) if not config.wavenet_natural_eval else None

						if tower_c[i] is not None:
							tower_c[i] = tf.expand_dims(tower_c[i][idx, :, :length], axis=0)
							with tf.control_dependencies([tf.assert_equal(tf.rank(tower_c[i]), 3)]):
								tower_c[i] = tf.identity(tower_c[i], name='eval_assert_c_rank_op')
								# print("tower_c[i] : {}".format(tower_c[i]))

						if tower_g[i] is not None:
							tower_g[i] = tf.expand_dims(tower_g[i][idx], axis=0)

						batch_size = tf.shape(tower_c[i])[0]

						#Start silence frame
						if is_mulaw_quantize(config.input_type):
							initial_value = mulaw_quantize(0, config.quantize_channels)
						elif is_mulaw(config.input_type):
							initial_value = mulaw(0.0, config.quantize_channels)
						else:
							initial_value = 0.0

						#[channels, ]
						if is_mulaw_quantize(config.input_type):
							initial_input = tf.one_hot(indices=initial_value, depth=config.quantize_channels, dtype=tf.float32)
							initial_input = tf.tile(tf.reshape(initial_input, [1, 1, config.quantize_channels]), [batch_size, 1, 1])
						else:
							initial_input = tf.ones([batch_size, 1, 1], tf.float32) * initial_value

						#Fast eval
						y_hat = self.incremental(initial_input, c=tower_c[i], g=tower_g[i], time_length=length, test_inputs=test_inputs,
							softmax=False, quantize=True, log_scale_min=config.log_scale_min, log_scale_min_gauss=config.log_scale_min_gauss)

						#Save targets and length for eval loss computation
						if is_mulaw_quantize(config.input_type):
							self.tower_y_eval.append(tf.reshape(y[idx], [1, -1])[:, :length])
						else:
							self.tower_y_eval.append(tf.expand_dims(y[idx], axis=0)[:, :length, :])
						self.tower_eval_length.append(length)

						if is_mulaw_quantize(config.input_type):
							y_hat = tf.reshape(tf.argmax(y_hat, axis=1), [-1])
							y_hat = inv_mulaw_quantize(y_hat, config.quantize_channels)
							y_target = inv_mulaw_quantize(y_target, config.quantize_channels)
						elif is_mulaw(config.input_type):
							y_hat = inv_mulaw(tf.reshape(y_hat, [-1]), config.quantize_channels)
							y_target = inv_mulaw(y_target, config.quantize_channels)
						else:
							y_hat = tf.reshape(y_hat, [-1])

						self.tower_y_hat.append(y_hat)
						self.tower_y_target.append(y_target)
						self.tower_eval_c.append(tower_c[i][idx])
						self.tower_eval_upsampled_local_features.append(self.upsampled_local_features[idx])

						if self.local_conditioning_enabled():
							log('  local_condition:           {}'.format(tower_c[i].shape))
						if self.has_speaker_embedding():
							log('  global_condition:          {}'.format(tower_g[i].shape))
						log('  targets:                   {}'.format(y_target.shape))
						log('  outputs:                   {}'.format(y_hat.shape))

					#synthesizing
					else:
						batch_size = tf.shape(tower_c[i])[0]
						if c is None:
							assert synthesis_length is not None
						else:
							#[batch_size, local_condition_time, local_condition_dimension(num_mels)]
							message = ('Expected 3 dimension shape [batch_size(1), time_length, {}] for local condition features but found {}'.format(
									config.cin_channels, tower_c[i].shape))
							with tf.control_dependencies([tf.assert_equal(tf.rank(tower_c[i]), 3, message=message)]):
								tower_c[i] = tf.identity(tower_c[i], name='synthesis_assert_c_rank_op')

							Tc = tf.shape(tower_c[i])[1] #num_time_frames of mfcc
							upsample_factor = audio.get_hop_size(self._config) #160


							#Overwrite length with respect to local condition features
							#### Keep synthesis_length
							#synthesis_length = Tc * upsample_factor
							synthesis_length = tf.cast(tf.math.ceil(Tc / 4) * 4 * upsample_factor, tf.int32)

							#[batch_size, local_condition_dimension, local_condition_time]
							#time_length will be corrected using the upsample network
							#tower_c[i] = tf.transpose(tower_c[i], [0, 2, 1])

						#if tower_g[i] is not None:

							#assert tower_g[i].shape == (batch_size, 1)


						#Start silence frame
						if is_mulaw_quantize(config.input_type):
							initial_value = mulaw_quantize(0, config.quantize_channels)
						elif is_mulaw(config.input_type):
							initial_value = mulaw(0.0, config.quantize_channels)
						else:
							initial_value = 0.0

						if is_mulaw_quantize(config.input_type):
							assert initial_value >= 0 and initial_value < config.quantize_channels
							initial_input = tf.one_hot(indices=initial_value, depth=config.quantize_channels, dtype=tf.float32)
							initial_input = tf.tile(tf.reshape(initial_input, [1, 1, config.quantize_channels]), [batch_size, 1, 1])
						else:
							initial_input = tf.ones([batch_size, 1, 1], tf.float32) * initial_value

						y_hat = self.incremental(initial_input, c=tower_c[i], g=tower_g[i], time_length=synthesis_length, test_inputs=tower_test_inputs[i],
							softmax=False, quantize=True, log_scale_min=config.log_scale_min, log_scale_min_gauss=config.log_scale_min_gauss)

						if is_mulaw_quantize(config.input_type):
							y_hat = tf.reshape(tf.argmax(y_hat, axis=1), [batch_size, -1])
							y_hat = util.inv_mulaw_quantize(y_hat, config.quantize_channels)
						elif is_mulaw(config.input_type):
							y_hat = util.inv_mulaw(tf.reshape(y_hat, [batch_size, -1]), config.quantize_channels)
						else:
							y_hat = tf.reshape(y_hat, [batch_size, -1])

						self.tower_y_hat.append(y_hat)
						self.tower_synth_upsampled_local_features.append(self.upsampled_local_features)

						if self.local_conditioning_enabled():
							log('  local_condition:           {}'.format(tower_c[i].shape))
						if self.has_speaker_embedding():
							log('  global_condition:          {}'.format(tower_g[i].shape))
						log('  outputs:                   {}'.format(y_hat.shape))

		#self.variables = tf.trainable_variables()

		# Check trainable variables
		all_train_vars = tf.trainable_variables()
		# print("all_train_vars : {}\n{}".format(len(all_train_vars), all_train_vars))
		vqvae_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="WaveNet_model/inference/vqencoder")
		# print("vqvae_train_vars : {}\n{}".format(len(vqvae_train_vars), vqvae_train_vars))
		wavenet_train_vars =  list(set(all_train_vars) - set(vqvae_train_vars))

		spk_train_vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="WaveNet_model/inference/speaker_classifier") #spkclassfier
		spk_train_vars = vqvae_train_vars + spk_train_vars_
		# print("spk_train_vars : {}\n{}".format(len(spk_train_vars), spk_train_vars))
		# print("wavenet_train_vars : {}\n{}".format(len(wavenet_train_vars), wavenet_train_vars))
		encdec_train_vars = list(set(all_train_vars) - set(spk_train_vars))

		if config.post_train_mode == True:       #post #yk: post_train일 경우 variable을 wavenet vars 지정
			self.variables = wavenet_train_vars
		elif config.pre_train_mode == True : 	#spkclassfier
			self.variables = encdec_train_vars
		elif config.spk_train_mode == True : 	#spkclassfier
			self.variables = spk_train_vars
		else:	 # train all vars
			self.variables = all_train_vars

		log('  Receptive Field:           ({} samples / {:.1f} ms)'.format(self.receptive_field, self.receptive_field / config.sample_rate * 1000.))
		#1_000_000 is causing syntax problems for some people?! Python please :)
		log('  WaveNet Parameters:        {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.variables]) / 1000000))

		self.ema = tf.train.ExponentialMovingAverage(decay=config.wavenet_ema_decay)


	def add_loss(self, global_step):
		'''Adds loss computation to the graph. Supposes that initialize function has already been called.
		'''
		self.tower_loss = []
		total_loss = 0
		gpus = ['/gpu:{}'.format(i) for i in range(self._config.wavenet_num_gpus)]

		all_train_lambda = tf.cond(global_step-tf.convert_to_tensor(self._config.pre_train_steps)-tf.convert_to_tensor(self._config.spk_train_steps) < tf.convert_to_tensor(50000) ,\
		 lambda: tf.convert_to_tensor(0.0000002, dtype=tf.float32)*(tf.cast(global_step, dtype=tf.float32)-tf.convert_to_tensor(self._config.pre_train_steps, dtype=tf.float32)-tf.convert_to_tensor(self._config.spk_train_steps, dtype=tf.float32)), \
		 lambda: tf.convert_to_tensor(0.01, dtype=tf.float32))
		print("all_train_lambda : {}".format(all_train_lambda)) 

		for i in range(self._config.wavenet_num_gpus):
			with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device='/cpu:0', worker_device=gpus[i])):
				with tf.variable_scope('loss') as scope:
					if self.is_training:
						if is_mulaw_quantize(self._config.input_type):
							tower_loss = MaskedCrossEntropyLoss(self.tower_y_hat_q[i][:, :-1, :], self.tower_y[i][:, 1:], mask=self.tower_mask[i])
						else:
							if self._config.out_channels == 2:
								tower_loss = GaussianMaximumLikelihoodEstimation(self.tower_y_hat_train[i][:, :, :-1], self.tower_y[i][:, 1:, :],
									config=self._config, mask=self.tower_mask[i])
							else:
								tower_loss = DiscretizedMixtureLogisticLoss(self.tower_y_hat_train[i][:, :, :-1], self.tower_y[i][:, 1:, :],
									config=self._config, mask=self.tower_mask[i])

						# if self._config.spk_train_mode == True : #spkclassifier
							#One hot encode targets (outputs.shape[-1] = config.quantize_channels)
							# tower_g_onehot = tf.one_hot(self.tower_g[i], depth=tf.shape(self.tower_spk_hat_train[i])[-1])
						tower_g_onehot = tf.reshape(tf.one_hot(self.tower_g[i], depth=self._config.n_speakers), [-1, self._config.n_speakers])

						print("self.tower_g[i]: {}".format(self.tower_g[i]))
						print("self.tower_spk_hat_train[i]: {}".format(self.tower_spk_hat_train[i]))
						print("tower_g_onehot: {}".format(tower_g_onehot))

						with tf.control_dependencies([tf.assert_equal(tf.shape(self.tower_spk_hat_train[i]), tf.shape(tower_g_onehot))]):
							spk_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.tower_spk_hat_train[i], labels=tower_g_onehot))

					elif self.is_evaluating:
						if is_mulaw_quantize(self._config.input_type):
							tower_loss = MaskedCrossEntropyLoss(self.tower_y_hat_eval[i], self.tower_y_eval[i], lengths=[self.tower_eval_length[i]])
						else:
							if self._config.out_channels == 2:
								tower_loss = GaussianMaximumLikelihoodEstimation(self.tower_y_hat_eval[i], self.tower_y_eval[i],
									config=self._config, lengths=[self.tower_eval_length[i]])
							else:
								tower_loss = DiscretizedMixtureLogisticLoss(self.tower_y_hat_eval[i], self.tower_y_eval[i],
									config=self._config, lengths=[self.tower_eval_length[i]])
						# if self._config.spk_train_mode == True : #spkclassifier
						spk_loss = tf.constant(0, dtype=tf.float32)

					else:
						raise RuntimeError('Model not in train/eval mode but computing loss: Where did this go wrong?')

			#Compute final loss
			if self._config.post_train_mode == True:
				self.tower_loss.append(tower_loss)
				total_loss += tower_loss
			elif self._config.pre_train_mode == True :
				self.tower_loss.append(tower_loss + self.vq_loss)
				total_loss += (tower_loss + self.vq_loss)
			elif self._config.spk_train_mode == True :
				self.tower_loss.append(spk_loss + self.vq_loss)
				total_loss += (spk_loss + self.vq_loss)
			else :
				self.tower_loss.append(tower_loss + self.vq_loss - all_train_lambda * spk_loss)
				total_loss += (tower_loss + self.vq_loss - all_train_lambda * spk_loss)

		if self.is_training:
			self.loss = total_loss / self._config.wavenet_num_gpus
			self.reconst_loss = tower_loss / self._config.wavenet_num_gpus
			self.spk_loss = spk_loss / self._config.wavenet_num_gpus

		else:
			self.eval_loss = total_loss / self._config.wavenet_num_gpus


	def add_optimizer(self, global_step, post_train_mode, pre_train_mode, spk_train_mode): #post #optimizer에 post_train_mode 파라미터 추가
		'''Adds optimizer to the graph. Supposes that initialize function has already been called.
		'''
		hp = self._config
		tower_gradients = []

		# Check trainable variables
		self.all_train_vars = tf.trainable_variables()
		# print("all_train_vars : {}\n{}".format(len(all_train_vars), all_train_vars))
		vqvae_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="WaveNet_model/inference/vqencoder")
		# print("vqvae_train_vars : {}\n{}".format(len(vqvae_train_vars), vqvae_train_vars))
		wavenet_train_vars =  list(set(self.all_train_vars) - set(vqvae_train_vars))
		# print("wavenet_train_vars : {}\n{}".format(len(wavenet_train_vars), wavenet_train_vars))
		spk_train_vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="WaveNet_model/inference/speaker_classifier") #spkclassfier
		spk_train_vars = vqvae_train_vars + spk_train_vars_
		encdec_train_vars = list(set(self.all_train_vars) - set(spk_train_vars))

		# 1. Declare GPU devices
		gpus = ['/gpu:{}'.format(i) for i in range(hp.wavenet_num_gpus)]

		grad_device = '/cpu:0' if hp.wavenet_num_gpus > 1 else gpus[0]

		with tf.device(grad_device):
			with tf.variable_scope('optimizer'):
				#Create lr schedule
				if hp.wavenet_lr_schedule == 'noam':
					learning_rate = self._noam_learning_rate_decay(hp.wavenet_learning_rate,
						global_step,
						warmup_steps=hp.wavenet_warmup)
				elif hp.wavenet_lr_schedule == 'exponential':
					learning_rate = self._exponential_learning_rate_decay(hp.wavenet_learning_rate,
						global_step,
						hp.wavenet_decay_rate,
						hp.wavenet_decay_steps)
				else:
					assert hp.wavenet_lr_schedule == 'piecewise'
					learning_rate = tf.train.piecewise_constant(global_step, hp.wavenet_lr_boundaries, hp.wavenet_lr_values, name='wavenet_piecewise_constant_decay')

				#Adam optimization
				self.learning_rate = learning_rate
				optimizer = tf.train.AdamOptimizer(learning_rate, hp.wavenet_adam_beta1,
					hp.wavenet_adam_beta2, hp.wavenet_adam_epsilon)

		# 2. Compute Gradient
		for i in range(hp.wavenet_num_gpus):
			#Device placemenet
			with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device='/cpu:0', worker_device=gpus[i])):
				with tf.variable_scope('optimizer') as scope:
					if hp.spk_train_mode == True:  # spk-train mode      #spkclassifier                                                          #post
						gradients = optimizer.compute_gradients(self.tower_loss[i], var_list=spk_train_vars)  #post #yk: wavenet_vars만 gradient 구함
						tower_gradients.append(gradients)
					elif hp.pre_train_mode == True:  #pre-train mode                                                              #post
						gradients = optimizer.compute_gradients(self.tower_loss[i], var_list=encdec_train_vars)  #post #yk: wavenet_vars만 gradient 구함
						tower_gradients.append(gradients)
					elif hp.post_train_mode == True:  # post-train mode                                                              #post
						gradients = optimizer.compute_gradients(self.tower_loss[i], var_list=wavenet_train_vars)  #post #yk: wavenet_vars만 gradient 구함
						tower_gradients.append(gradients)                             #post
					else :  # train all vars                                        	#post
						gradients = optimizer.compute_gradients(self.tower_loss[i])                 #post
						tower_gradients.append(gradients)                             #post

		# 3. Average Gradient
		with tf.device(grad_device):
			avg_grads = []
			variables = []
			for grad_and_vars in zip(*tower_gradients):
				# each_grads_vars = ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
				if grad_and_vars[0][0] is not None:
					grads = []
					for g, _ in grad_and_vars:
						expanded_g = tf.expand_dims(g, 0)
						#Append on a "tower" dimension which we will average over below.
						grads.append(expanded_g)

					#Average over the 'tower' dimension.
					grad = tf.concat(axis=0, values=grads)
					grad = tf.reduce_mean(grad, 0)
				else:
					grad = grad_and_vars[0][0]

				v = grad_and_vars[0][1]
				avg_grads.append(grad)
				variables.append(v)

			self.gradients = avg_grads

			#Gradients clipping
			if hp.wavenet_clip_gradients:
				#Clip each gradient by a [min, max] range of values and its norm by [0, max_norm_value]
				clipped_grads = []
				for g in avg_grads:
					if g is not None:
						clipped_g = tf.clip_by_norm(g, hp.wavenet_gradient_max_norm)
						clipped_g = tf.clip_by_value(clipped_g, -hp.wavenet_gradient_max_value, hp.wavenet_gradient_max_value)
						clipped_grads.append(clipped_g)

					else:
						clipped_grads.append(g)

			else:
				clipped_grads = avg_grads

			# if (post_train_mode == False) and (spk_train_mode == False) and (pre_train_mode == False) :  # train all vars     #spkclassfier                                    #post #yk:post-train일 경우 dependency check 없앰
			# 	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			# 		adam_optimize = optimizer.apply_gradients(zip(clipped_grads, variables), global_step=global_step)
			# else:
			# 	adam_optimize = optimizer.apply_gradients(zip(clipped_grads, variables),global_step=global_step)
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				adam_optimize = optimizer.apply_gradients(zip(clipped_grads, variables), global_step=global_step)

			#Add exponential moving average
			#https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
			#Use adam optimization process as a dependency
			with tf.control_dependencies([adam_optimize]):
				#Create the shadow variables and add ops to maintain moving averages
				#Also updates moving averages after each update step
				#This is the optimize call instead of traditional adam_optimize one.
				assert set(self.variables) == set(variables) #Verify all trainable variables are being averaged
				# self.optimize = self.ema.apply(variables)
				self.optimize = self.ema.apply(self.all_train_vars)

	def _noam_learning_rate_decay(self, init_lr, global_step, warmup_steps=4000.0):
		# Noam scheme from tensor2tensor:
		step = tf.cast(global_step + 1, dtype=tf.float32)
		return tf.maximum(init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5), 1e-4)

	def _exponential_learning_rate_decay(self, init_lr, global_step,
							 decay_rate=0.5,
							 decay_steps=300000):
		#Compute natural exponential decay
		lr = tf.train.exponential_decay(init_lr,
			global_step,
			decay_steps,
			decay_rate,
			name='wavenet_lr_exponential_decay')
		return lr


	def get_mask(self, input_lengths, maxlen=None):
		expand = not is_mulaw_quantize(self._config.input_type)
		mask = sequence_mask(input_lengths, max_len=maxlen, expand=expand)

		if is_mulaw_quantize(self._config.input_type):
			return mask[:, 1:]
		return mask[:, 1:, :]

	#Sanity check functions
	def has_speaker_embedding(self):
		return self.embed_speakers is not None

	def local_conditioning_enabled(self):
		return self._config.cin_channels > 0

	def global_conditioning_enabled(self):
		return self._config.gin_channels > 0

	def step(self, x, c=None, g=None, softmax=False):
		"""Forward step

		Args:
			x: Tensor of shape [batch_size, channels, time_length], One-hot encoded audio signal.
			c: Tensor of shape [batch_size, time_length, cin_channels], Local conditioning features.
			g: Tensor of shape [batch_size, gin_channels, 1] or *Ids of shape [batch_size, 1],*
				Global conditioning features.
				Note: set config.use_speaker_embedding to False to disable embedding layer and
				use extrnal One-hot encoded features.
			softmax: Boolean, Whether to apply softmax.

		Returns:
			a Tensor of shape [batch_size, out_channels, time_length]
		"""
		#[batch_size, channels, time_length] -> [batch_size, time_length, channels]
		batch_size = tf.shape(x)[0]
		time_length = tf.shape(x)[-1]

		if g is not None:
			if self.embed_speakers is not None:
				#[batch_size, 1] ==> [batch_size, 1, gin_channels]
				g = self.embed_speakers(tf.reshape(g, [batch_size, -1]))
				#[batch_size, gin_channels, 1]
				with tf.control_dependencies([tf.assert_equal(tf.rank(g), 3)]):
					g = tf.transpose(g, [0, 2, 1])

		#Expand global conditioning features to all time steps
		g_bct = _expand_global_features(batch_size, time_length, g, data_format='BCT')



		#Local Conditioning
		if c is not None:
			#### vq_out to self.vq_out
			vq_out = self.vqencoder(c)
			self.vq_loss = vq_out["loss"]
			self.vq_perplexity = vq_out["perplexity"]
			#### VQVAE Quantized Outputs
			self.vq_embeddings = vq_out["quantize"] #vqvae output shape [batch_size, time_steps, c_channels]
			# print("vq_embeddings: {}".format(self.vq_embeddings))

			c = tf.transpose(self.vq_embeddings, (0,2,1)) #[batch_size, c_channels, time_steps]
			# Speaker Classifier model 								#spkclassifier
			spk = self.spkclassifier(c) 			#spkclassifier
			# print("spk classifier output: {}".format(spk))

			c = tf.layers.dropout(c, rate=self._config.vqvae_dropout, noise_shape=[batch_size, 1, time_length])
			#First convolutional layer for local condition
			c = self.lc_first_conv(c)

			if self._config.upsample_type == '2D':
				#[batch_size, 1, cin_channels, time_length]
				expand_dim = 1
			elif self._config.upsample_type == '1D':
				#[batch_size, cin_channels, 1, time_length]
				expand_dim = 2
			else:
				assert self._config.upsample_type in ('Resize', 'SubPixel', 'NearestNeighbor')
				#[batch_size, cin_channels, time_length, 1]
				expand_dim = 3


			c = tf.expand_dims(c, axis=expand_dim)
			# print("local condition expanded: {}".format(c))

			for transposed_conv in self.upsample_conv:
				c = transposed_conv(c)
				# print("local condition trans_conv: {}".format(c))


			#[batch_size, cin_channels, time_length]
			c = tf.squeeze(c, [expand_dim])
			# print("local condition sqeeze: {}".format(c))

			with tf.control_dependencies([tf.assert_equal(tf.shape(c)[-1], tf.shape(x)[-1])]):
				c = tf.identity(c, name='control_c_and_x_shape')

			self.upsampled_local_features = c

		#Feed data to network
		x = self.first_conv(x)
		skips = None
		for conv in self.residual_layers:
			x, h = conv(x, c=c, g=g_bct)
			if skips is None:
				skips = h
			else:
				skips = skips + h

				if self._config.legacy:
					skips = skips * np.sqrt(0.5)
		x = skips

		for conv in self.last_conv_layers:
			x = conv(x)

		return tf.nn.softmax(x, axis=1) if softmax else x, spk


	def incremental(self, initial_input, c=None, g=None,
		time_length=100, test_inputs=None,
		softmax=True, quantize=True, log_scale_min=-7.0, log_scale_min_gauss=-7.0):
		"""Inceremental forward step

		Inputs of shape [batch_size, channels, time_length] are reshaped to [batch_size, time_length, channels]
		Input of each time step is of shape [batch_size, 1, channels]

		Args:
			Initial input: Tensor of shape [batch_size, channels, 1], initial recurrence input.
			c: Tensor of shape [batch_size, cin_channels, time_length], Local conditioning features
			g: Tensor of shape [batch_size, gin_channels, time_length] or [batch_size, gin_channels, 1]
				global conditioning features
			T: int, number of timesteps to generate
			test_inputs: Tensor, teacher forcing inputs (debug)
			softmax: Boolean, whether to apply softmax activation
			quantize: Whether to quantize softmax output before feeding to
				next time step input
			log_scale_min: float, log scale minimum value.

		Returns:
			Tensor of shape [batch_size, channels, time_length] or [batch_size, channels, 1]
				Generated one_hot encoded samples
		"""
		batch_size = tf.shape(initial_input)[0]
		# print("batch_size : {}".format(batch_size))

		#Note: should reshape to [batch_size, time_length, channels]
		#not [batch_size, channels, time_length]
		if test_inputs is not None:
			if self.scalar_input:
				if tf.shape(test_inputs)[1] == 1:
					test_inputs = tf.transpose(test_inputs, [0, 2, 1])
			else:
				test_inputs = tf.cast(test_inputs, tf.int32)
				test_inputs = tf.one_hot(indices=test_inputs, depth=self._config.quantize_channels, dtype=tf.float32)
				test_inputs = tf.squeeze(test_inputs, [2])

				if tf.shape(test_inputs)[1] == self._config.out_channels:
					test_inputs = tf.transpose(test_inputs, [0, 2, 1])

			batch_size = tf.shape(test_inputs)[0]
			if time_length is None:
				time_length = tf.shape(test_inputs)[1]
			else:
				time_length = tf.maximum(time_length, tf.shape(test_inputs)[1])

		#Global conditioning
		if g is not None:
			if self.embed_speakers is not None:
				g = self.embed_speakers(tf.reshape(g, [batch_size, -1]))
				#[batch_size, channels, 1]
				with tf.control_dependencies([tf.assert_equal(tf.rank(g), 3)]):
					g = tf.transpose(g, [0, 2, 1])

		self.g_btc = _expand_global_features(batch_size, time_length, g, data_format='BTC')

		#Local conditioning
		if c is not None:
			## vqvae
			# print("c original  : {}".format(c))
			c = tf.reshape(c, [batch_size, -1, self._config.num_mfccs])
			# print("c reshaped  : {}".format(c))
			vq_out = self.vqencoder(c)
			self.vq_loss = vq_out["loss"]
			self.vq_perplexity = vq_out["perplexity"]
			#### VQVAE Quantized Outputs
			self.vq_embeddings = vq_out["quantize_frozen"]
			self.vq_onehot = vq_out["encodings"]
			self.vq_w = vq_out["wmatrix"]
			self.vq_enc_ind = vq_out["encoding_indices"]

			# self.vq_embeddings = vq_out["quantize"] #vqvae output shape [batch_size, time_steps, c_channels]
			# print("self.vq_embeddings : {}".format(self.vq_embeddings))
			c = tf.transpose(self.vq_embeddings, (0,2,1)) #[batch_size, c_channels, time_steps]
			# print("c transposed  : {}".format(c))

			 #First convolutional layer for local condition
			self.lc_first_conv.training = True
			c = self.lc_first_conv(c)
			# print("c first conv  : {}".format(c))

			if self._config.upsample_type == '2D':
				#[batch_size, 1, cin_channels, time_length]
				expand_dim = 1
			elif self._config.upsample_type == '1D':
				#[batch_size, cin_channels, 1, time_length]
				expand_dim = 2
			else:
				assert self._config.upsample_type in ('Resize', 'SubPixel', 'NearestNeighbor')
				#[batch_size, cin_channels, time_length, 1]
				expand_dim = 3

			c = tf.expand_dims(c, axis=expand_dim)

			for upsample_conv in self.upsample_conv:
				c = upsample_conv(c)

			#[batch_size, channels, time_length]
			c = tf.squeeze(c, [expand_dim])

			##### Debug / self.c must have shape BTC
			#with tf.control_dependencies([tf.assert_equal(tf.shape(c)[-1], time_length)]):
			#	self.c = tf.transpose(c, [0, 2, 1])
			self.c = tf.transpose(c, [0, 2, 1])


			self.upsampled_local_features = c


		#Initialize loop variables
		if initial_input.shape[1] == self._config.out_channels:
			initial_input = tf.transpose(initial_input, [0, 2, 1])

		initial_time = tf.constant(0, dtype=tf.int32)
		# if test_inputs is not None:
		# 	initial_input = tf.expand_dims(test_inputs[:, 0, :], axis=1)
		initial_outputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		initial_loss_outputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
		#Only use convolutions queues for Residual Blocks main convolutions (only ones with kernel size 3 and dilations, all others are 1x1)
		initial_queues = [tf.zeros((batch_size, res_conv.layer.kw + (res_conv.layer.kw - 1) * (res_conv.layer.dilation_rate[0] - 1), self._config.residual_channels),
			name='convolution_queue_{}'.format(i+1)) for i, res_conv in enumerate(self.residual_layers)]

		def condition(time, unused_outputs_ta, unused_current_input, unused_loss_outputs_ta, unused_queues):
			return tf.less(time, time_length)

		def body(time, outputs_ta, current_input, loss_outputs_ta, queues):
			#conditioning features for single time step
			ct = None if self.c is None else tf.expand_dims(self.c[:, time, :], axis=1)
			gt = None if self.g_btc is None else tf.expand_dims(self.g_btc[:, time, :], axis=1)

			x = self.first_conv.incremental_step(current_input)

			skips = None
			new_queues = []
			for conv, queue in zip(self.residual_layers, queues):
				x, h, new_queue = conv.incremental_step(x, c=ct, g=gt, queue=queue)

				if self._config.legacy:
					skips = h if skips is None else (skips + h) * np.sqrt(0.5)
				else:
					skips = h if skips is None else (skips + h)
				new_queues.append(new_queue)
			x = skips

			for conv in self.last_conv_layers:
				try:
					x = conv.incremental_step(x)
				except AttributeError: #When calling Relu activation
					x = conv(x)

			#Save x for eval loss computation
			loss_outputs_ta = loss_outputs_ta.write(time, tf.squeeze(x, [1])) #squeeze time_length dimension (=1)

			#Generate next input by sampling
			if self.scalar_input:
				if self._config.out_channels == 2:
					x = sample_from_gaussian(
						tf.reshape(x, [batch_size, -1, 1]),
						log_scale_min_gauss=log_scale_min_gauss)
				else:
					x = sample_from_discretized_mix_logistic(
						tf.reshape(x, [batch_size, -1, 1]), log_scale_min=log_scale_min)

				next_input = tf.expand_dims(x, axis=-1) #Expand on the channels dimension
			else:
				x = tf.nn.softmax(tf.reshape(x, [batch_size, -1]), axis=1) if softmax \
					else tf.reshape(x, [batch_size, -1])
				if quantize:
					#[batch_size, 1]
					sample = tf.multinomial(x, 1) #Pick a sample using x as probability (one for each batch)
					#[batch_size, 1, quantize_channels] (time dimension extended by default)
					x = tf.one_hot(sample, depth=self._config.quantize_channels)

				next_input = x

			if len(x.shape) == 3:
				x = tf.squeeze(x, [1])

			outputs_ta = outputs_ta.write(time, x)

			#Override input with ground truth
			if test_inputs is not None:
				next_input = tf.expand_dims(test_inputs[:, time, :], axis=1)

			time = tf.Print(time + 1, [time+1, time_length])
			#output = x (maybe next input)
			# if test_inputs is not None:
			# 	#override next_input with ground truth
			# 	next_input = tf.expand_dims(test_inputs[:, time, :], axis=1)

			return (time, outputs_ta, next_input, loss_outputs_ta, new_queues)

		res = tf.while_loop(
			condition,
			body,
			loop_vars=[
				initial_time, initial_outputs_ta, initial_input, initial_loss_outputs_ta, initial_queues
			],
			parallel_iterations=32,
			swap_memory=self._config.wavenet_swap_with_cpu)

		outputs_ta = res[1]
		#[time_length, batch_size, channels]
		outputs = outputs_ta.stack()

		#Save eval prediction for eval loss computation
		eval_outputs = res[3].stack()

		self.tower_y_hat_eval = []
		if is_mulaw_quantize(self._config.input_type):
			self.tower_y_hat_eval.append(tf.transpose(eval_outputs, [1, 0, 2]))
		else:
			self.tower_y_hat_eval.append(tf.transpose(eval_outputs, [1, 2, 0]))

		#[batch_size, channels, time_length]
		return tf.transpose(outputs, [1, 2, 0])

	def clear_queue(self):
		self.first_conv.clear_queue()
		for f in self.conv_layers:
			f.clear_queue()
		for f in self.last_conv_layers:
			try:
				f.clear_queue()
			except AttributeError:
				pass



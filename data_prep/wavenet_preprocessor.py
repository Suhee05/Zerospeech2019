import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from utils import audio
from utils.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize


def build_from_path(hparams, input_dir, mfcc_dir, wav_dir, n_jobs=12, mode='train', tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mfcc_dir: output directory of the preprocessed speech mfcc dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []

	for file in os.listdir(input_dir):
		wav_path = os.path.join(input_dir, file)
		basename = os.path.basename(wav_path).replace('.wav', '')
		futures.append(executor.submit(partial(_process_utterance, mfcc_dir, wav_dir, basename, wav_path, hparams, mode)))


	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(mfcc_dir, wav_dir, index, wav_path, hparams, mode):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mfcc to disk and return a tuple to write
	to the train.txt file

	Args:
		- mfcc_dir: the directory to write the mfcc into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectrogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mfcc_filename, linear_filename, time_steps, mfcc_frames, linear_frames, text)
	"""

	try:
		# Load the audio as numpy array
		wav_full = audio.load_wav(wav_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
			wav_path))
		return None

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav_full = audio.trim_silence(wav_full, hparams)

	# Preprocess Audio & Extract MFCC (mfcc + d + a)
	sample_idx = 0
	sample_metadata = []

	if (mode == "train") or (mode == "post_train"):
		# Add the same size slice from the end
		if wav_full.shape[0] >= hparams.sample_size:
			n_slice = int(np.floor(wav_full.shape[0]/hparams.sample_size))
			samples = wav_full[:n_slice * hparams.sample_size].reshape((n_slice, hparams.sample_size))
			if wav_full.shape[0] % hparams.sample_size != 0:
				## FOR UNIT SEARCH : slice each audio by sample_size
				last_slice = wav_full[::-1][:hparams.sample_size]
				samples = np.vstack((samples, last_slice))
		else:
			samples = [wav_full]
	else:
		samples = [wav_full]


	for wav in samples:

		#Pre-emphasize
		preem_wav = audio.preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

		#rescale wav
		if hparams.rescale:
			wav = wav / np.abs(wav).max() * hparams.rescaling_max
			preem_wav = preem_wav / np.abs(preem_wav).max() * hparams.rescaling_max

			#Assert all audio is in [-1, 1]
			if (wav > 1.).any() or (wav < -1.).any():
				raise RuntimeError('wav has invalid value: {}'.format(wav_path))
			if (preem_wav > 1.).any() or (preem_wav < -1.).any():
				raise RuntimeError('wav has invalid value: {}'.format(wav_path))

		#Mu-law quantize
		if is_mulaw_quantize(hparams.input_type):
			#[0, quantize_channels)
			out = mulaw_quantize(wav, hparams.quantize_channels)

			# #Trim silences
			# start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
			# wav = wav[start: end]
			# preem_wav = preem_wav[start: end]
			# out = out[start: end]

			constant_values = mulaw_quantize(0, hparams.quantize_channels)
			out_dtype = np.int16

		elif is_mulaw(hparams.input_type):
			#[-1, 1]
			out = mulaw(wav, hparams.quantize_channels)
			constant_values = mulaw(0., hparams.quantize_channels)
			out_dtype = np.float32

		else:
			#[-1, 1]
			out = wav
			constant_values = 0.
			out_dtype = np.float32

		# Compute mfcc
		mfcc = audio.mfcc(wav, hparams)
		mfcc_frames = mfcc.shape[0]

		# # Compute the mel scale spectrogram from the wav
		# mel_spectrogram = audio.melspectrogram(preem_wav, hparams).astype(np.float32)
		# mel_frames = mel_spectrogram.shape[1]

		if mfcc_frames > hparams.max_mel_frames and hparams.clip_mels_length:
			return None

		#Ensure time resolution adjustement between audio and mel-spectrogram
		l_pad, r_pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams))
		#Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
		out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

		assert len(out) >= mfcc_frames * audio.get_hop_size(hparams)

		#time resolution adjustement
		#ensure length of raw audio is multiple of hop size so that we can use
		out = out[:int(np.ceil(mfcc_frames/hparams.vqvae_down_freq) * hparams.vqvae_down_freq * audio.get_hop_size(hparams))]
		assert len(out) % audio.get_hop_size(hparams) == 0
		time_steps = len(out)

		# Write the spectrogram and audio to disk
		audio_filename = os.path.join(wav_dir, 'audio-{}-{}.npy'.format(index, sample_idx))
		mfcc_filename = os.path.join(mfcc_dir, 'mfcc-{}-{}.npy'.format(index, sample_idx))
		np.save(audio_filename, out.astype(out_dtype), allow_pickle=False)
		np.save(mfcc_filename, mfcc, allow_pickle=False)

		#global condition features
		if hparams.gin_channels > 0:
			if (mode == "train") or (mode == "post_train"):
				speaker_id = hparams.speakers.index(index[:4])
			elif mode == "synth":
				speaker_id = 0
			else:
				speaker_id = '<no_g>'

		sample_metadata.append((audio_filename, mfcc_filename, mfcc_filename, speaker_id, time_steps, mfcc_frames))
		sample_idx += 1


	return sample_metadata

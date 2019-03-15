import argparse
import os
from multiprocessing import cpu_count

from data_prep import wavenet_preprocessor
from hparams import hparams
from tqdm import tqdm


def preprocess(args, input_dir, out_dir, hparams):
	mfcc_dir = os.path.join(out_dir, 'mfccs')
	wav_dir = os.path.join(out_dir, 'audio')
	os.makedirs(mfcc_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	metadata = wavenet_preprocessor.build_from_path(hparams, input_dir, mfcc_dir, wav_dir, args.n_jobs, args.mode, tqdm=tqdm)
	tmp = []
	while metadata:
		tmp.extend(metadata.pop(0))
	metadata = tmp
	write_metadata(metadata, out_dir, args.mode)
	print("metadata written")

def write_metadata(metadata, out_dir, mode):
	if mode == "synth":
		synth_dict = load_synthesis_dict()
	with open(os.path.join(out_dir, 'map.txt'), 'w', encoding='utf-8') as f:
		for m in metadata: # (audio_filename, mfcc_filename, mfcc_filename, speaker_id, time_steps, mfcc_frames)
			if mode == "synth":
				wav_name = os.path.basename(m[0]).split('-')[1] # audio-S122_3656689518-0.npy
				if wav_name in synth_dict.keys(): #S122_3656689518
					spk_id = synth_dict[wav_name] #V001
					m = list(m)
					m[3] = hparams.speakers.index(spk_id) # wavs to be generated in either two speakers in voice data
					print('audio file {} is in synthesis.txt. so speaker id was converted'.format(wav_name))

			f.write('|'.join([str(x) for x in m]) + '\n')

	mfcc_frames = sum([int(m[5]) for m in metadata])
	timesteps = sum([int(m[4]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), timesteps, hours))
	print('Max mfcc frames length: {}'.format(max(int(m[5]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[4] for m in metadata)))


def load_synthesis_dict():
	# Load synthesis.txt in dict
	synth_dict = {}
	with open(os.path.join(hparams.synth_input_dir, hparams.synth_language, 'synthesis.txt')) as f:
		wav_lines = f.read().splitlines()
	for wav_line in wav_lines:
		wav_name, wav_spk = wav_line.split(' ')
		if hparams.synth_language == "english":
			wav_name = wav_name.split("/")[-1]
		synth_dict[wav_name] = wav_spk # wav_name (ex) test/S079_... in eng, S378... in surprise
	return synth_dict


def run_preprocess(args, hparams):

	if args.mode == "train":
		preprocess(args, hparams.train_input_dir, hparams.wavenet_input, hparams)
	elif args.mode == "post_train":   #yk: post_train 일 경우 옵션 추가 (처리방식은 train 과 동일하고 input_dir 만 post_train_input_dir로 교체 )
		preprocess(args, hparams.post_train_input_dir, hparams.post_train_input, hparams)
	elif args.mode == "synth":
		preprocess(args, os.path.join(hparams.synth_input_dir, hparams.synth_language, 'test'), hparams.wavenet_synth, hparams)
	else:
		raise("Not supported mode")

def main():
	print('initśalizing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='train', required=True) #[train, post_train,synth]
	parser.add_argument('--n_jobs', type=int, default=cpu_count())

	args = parser.parse_args()

	run_preprocess(args, hparams)

if __name__ == '__main__':
	main()

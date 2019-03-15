import argparse
import os
from warnings import warn
from time import sleep
import numpy as np
import tensorflow as tf
from hparams import hparams, hparams_debug_string
from utils.infolog import log
from tqdm import tqdm
from wavenet.synthesizer import Synthesizer
from preprocess import load_synthesis_dict


def run_synthesis(checkpoint_path, output_dir, hparams):
    log_dir = os.path.join(output_dir, 'plots')
    wav_dir = os.path.join(output_dir, 'wavs')
    embed_dir = os.path.join(output_dir, 'embeddings')


    #We suppose user will provide correct folder depending on training method
    log(hparams_debug_string())
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams)

    metadata_filename = os.path.join(hparams.wavenet_synth, 'map.txt')
    with open(metadata_filename, encoding='utf-8') as f:
        metadata = np.array([line.strip().split('|') for line in f])
        if (hparams.synth_mode == "all") and (hparams.synth_idx != None):
            # if synth mode is all and synth_idx is not None, extract a part of metadata
            metadata = metadata[hparams.synth_idx[0]:hparams.synth_idx[1], :]


    # speaker ids from trained speakers list
    speaker_ids = metadata[:, 3]
    print("spk_ids" +str(speaker_ids.shape))
    mel_files = metadata[:, 1]
    print("mel_files" +str(mel_files.shape))

    log('Starting synthesis! (this will take a while..)')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)

    synth_dict = load_synthesis_dict()

    for idx, mel_file in enumerate(tqdm(mel_files)):
        print("idx")
        print(idx)
        mel_spectro = [np.load(mel_file)]
        basenames = [os.path.basename(mel_file).replace('.npy', '')]
        speaker_id = [speaker_ids[idx]]
        print("synthesizing {}".format(basenames[0]))

        if hparams.synth_mode == "all":
            if basenames[0].split('-')[1] in synth_dict.keys():
                print("Synthesizing both wav and embedding")
                synth.synthesize(mel_spectro, speaker_id, basenames, wav_dir, log_dir, embed_dir, embed_only=False)
            else:
                print("Synthesizing embedding only")
                synth.synthesize(mel_spectro, speaker_id, basenames, wav_dir, log_dir, embed_dir, embed_only=True)
        elif hparams.synth_mode == "embedding":
            print("Synthesizing embedding only")
            synth.synthesize(mel_spectro, speaker_id, basenames, wav_dir, log_dir, embed_dir, embed_only=True)
        elif hparams.synth_mode == "wav":
            if basenames[0].split('-')[1] in synth_dict.keys():
                synth.synthesize(mel_spectro, speaker_id, basenames, wav_dir, log_dir, embed_dir, embed_only=False)
        else:
            print("Not supported synth mode.")




    log('synthesized audio waveforms at {}'.format(wav_dir))



def wavenet_synthesize(hparams, checkpoint):
	output_dir = hparams.synth_output_dir

	try:
		checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
		log('loaded model at {}'.format(checkpoint_path))
	except:
		raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

	run_synthesis(checkpoint_path, output_dir, hparams)


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    wave_checkpoint = os.path.join(hparams.log_dir, 'wave_pretrained')
    wavenet_synthesize(hparams, wave_checkpoint)


if __name__ == '__main__':
    main()



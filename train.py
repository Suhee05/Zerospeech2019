import argparse
import os
import sys
import time
import traceback
from datetime import datetime
import json
import librosa
import numpy as np
import tensorflow as tf
from logger import log
from utils.audio import save_wavenet_wav, mfcc
import utils.util as util
from utils.valuewindow import ValueWindow
from feeder import Feeder, _interp
from hparams import hparams, hparams_debug_string
from wavenet.wavenet import WaveNet
from warnings import warn


def create_model(name, config, init=False):
    if util.is_mulaw_quantize(config.input_type):
        if config.out_channels != config.quantize_channels:
            raise RuntimeError(
                "out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")

    if name == 'WaveNet':
        return WaveNet(config, init)
    else:
        raise Exception('Unknow model: {}'.format(name))


# Logger
log = log()

def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

def add_embedding_stats(summary_writer, embedding_names, paths_to_meta, checkpoint_path):
    #Create tensorboard projector
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    config.model_checkpoint_path = checkpoint_path

    for embedding_name, path_to_meta in zip(embedding_names, paths_to_meta):
        #Initialize config
        embedding = config.embeddings.add()
        #Specifiy the embedding variable and the metadata
        embedding.tensor_name = embedding_name
        embedding.metadata_path = path_to_meta

    #Project the embeddings to space dimensions for visualization
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)

def add_train_stats(model, config):
    with tf.variable_scope('stats') as scope:
        for i in range(config.wavenet_num_gpus):
            tf.summary.histogram('wav_outputs %d' % i, model.tower_y_hat_log[i])
            tf.summary.histogram('wav_targets %d' % i, model.tower_y_log[i])
            if model.tower_means[i] is not None:
                tf.summary.histogram('gaussian_means %d' % i, model.tower_means[i])
                tf.summary.histogram('gaussian_log_scales %d' % i, model.tower_log_scales[i])

        tf.summary.scalar('wavenet_learning_rate', model.learning_rate)
        tf.summary.scalar('wavenet_loss', model.loss)

        gradient_norms = [tf.norm(grad) for grad in model.gradients if grad is not None]
        tf.summary.histogram('gradient_norm', gradient_norms)
        tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)) #visualize gradients (in case of explosion)
        return tf.summary.merge_all()

def add_test_stats(summary_writer, step, eval_loss, config):
    values = [
    tf.Summary.Value(tag='Wavenet_eval_model/eval_stats/wavenet_eval_loss', simple_value=eval_loss),
    ]

    test_summary = tf.Summary(value=values)
    summary_writer.add_summary(test_summary, step)


def create_shadow_saver(model, global_step=None):
    '''Load shadow variables of saved model.

    Inspired by: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    Can also use: shadow_dict = model.ema.variables_to_restore()
    '''
    #Add global step to saved variables to save checkpoints correctly

    shadow_variables = [model.ema.average_name(v) for v in model.all_train_vars]
    variables = model.all_train_vars

    if global_step is not None:
        shadow_variables += ['global_step']
        variables += [global_step]

    shadow_dict = dict(zip(shadow_variables, variables))
    return tf.train.Saver(shadow_dict, max_to_keep=20)

def load_averaged_model(sess, sh_saver, checkpoint_path):
    sh_saver.restore(sess, checkpoint_path)


def eval_step(sess, global_step, model, plot_dir, wav_dir, summary_writer, config):
    '''Evaluate model during training.
    Supposes that model variables are averaged.
    '''
    start_time = time.time()

    #### Vq Embedding

    y_hat, y_target, loss, input_mfcc, upsampled_features = sess.run([model.tower_y_hat[0], model.tower_y_target[0],
        model.eval_loss, model.tower_eval_c[0], model.tower_eval_upsampled_local_features[0]])
    duration = time.time() - start_time
    log.info('Time Evaluation: Generation of {} audio frames took {:.3f} sec ({:.3f} frames/sec)'.format(
        len(y_target), duration, len(y_target)/duration))

    #Make audio and plot paths
    pred_wav_path = os.path.join(wav_dir, 'step-{}-pred.wav'.format(global_step))
    target_wav_path = os.path.join(wav_dir, 'step-{}-real.wav'.format(global_step))
    plot_path = os.path.join(plot_dir, 'step-{}-waveplot.png'.format(global_step))
    mfcc_path = os.path.join(plot_dir, 'step-{}-reconstruction-mfcc.png'.format(global_step))
    upsampled_path = os.path.join(plot_dir, 'step-{}-upsampled-features.png'.format(global_step))


    #Save figure
    util.waveplot(plot_path, y_hat, y_target, model._config, title='{}, step={}, loss={:.5f}'.format(time_string(), global_step, loss))
    log.info('Eval loss for global step {}: {:.3f}'.format(global_step, loss))

    #Compare generated wav mel with original input mel to evaluate wavenet audio reconstruction performance
    #Both mels should match on low frequency information, wavenet mel should contain more high frequency detail when compared to Tacotron mels.
    T2_output_range = (-config.max_abs_value, config.max_abs_value) if config.symmetric_mels else (0, config.max_abs_value)
    generated_mfcc = _interp(mfcc(y_hat, config), T2_output_range)
    util.plot_spectrogram(generated_mfcc, mfcc_path, title='Local Condition vs Reconst. Mel-Spectrogram, step={}, loss={:.5f}'.format(
        global_step, loss), target_spectrogram=input_mfcc)
    util.plot_spectrogram(upsampled_features.T, upsampled_path, title='Upsampled Local Condition features, step={}, loss={:.5f}'.format(
        global_step, loss), auto_aspect=True)

    #Save Audio
    save_wavenet_wav(y_hat, pred_wav_path, sr=config.sample_rate, inv_preemphasize=config.preemphasize, k=config.preemphasis)
    save_wavenet_wav(y_target, target_wav_path, sr=config.sample_rate, inv_preemphasize=config.preemphasize, k=config.preemphasis)

    #Write eval summary to tensorboard
    log.info('Writing eval summary!')
    add_test_stats(summary_writer, global_step, loss, config=config)

def save_log(sess, global_step, model, plot_dir, wav_dir, config):
    log.info('\nSaving intermediate states at step {}'.format(global_step))
    idx = 0
    y_hat, y, loss, length, input_mfcc, upsampled_features = sess.run([model.tower_y_hat_log[0][idx],
        model.tower_y_log[0][idx],
        model.loss,
        model.tower_input_lengths[0][idx],
        model.tower_c[0][idx], model.tower_upsampled_local_features[0][idx]])

    #mask by length
    y_hat[length:] = 0
    y[length:] = 0

    #Make audio and plot paths
    pred_wav_path = os.path.join(wav_dir, 'step-{}-pred.wav'.format(global_step))
    target_wav_path = os.path.join(wav_dir, 'step-{}-real.wav'.format(global_step))
    plot_path = os.path.join(plot_dir, 'step-{}-waveplot.png'.format(global_step))
    mfcc_path = os.path.join(plot_dir, 'step-{}-reconstruction-mfcc.png'.format(global_step))
    upsampled_path = os.path.join(plot_dir, 'step-{}-upsampled-features.png'.format(global_step))

    #Save figure
    util.waveplot(plot_path, y_hat, y, config, title='{}, step={}, loss={:.5f}'.format(time_string(), global_step, loss))

    #Compare generated wav mfcc with original input mfcc to evaluate wavenet audio reconstruction performance
    T2_output_range = (-config.max_abs_value, config.max_abs_value) if config.symmetric_mels else (0, config.max_abs_value)
    generated_mfcc = _interp(mfcc(y_hat, config), T2_output_range)
    util.plot_spectrogram(generated_mfcc, mfcc_path, title='Local Condition vs Reconst. Mel-Spectrogram, step={}, loss={:.5f}'.format(
        global_step, loss), target_spectrogram=input_mfcc)
    util.plot_spectrogram(upsampled_features.T, upsampled_path, title='Upsampled Local Condition features, step={}, loss={:.5f}'.format(
        global_step, loss), auto_aspect=True)

    #Save audio
    save_wavenet_wav(y_hat, pred_wav_path, sr=config.sample_rate, inv_preemphasize=config.preemphasize, k=config.preemphasis)
    save_wavenet_wav(y, target_wav_path, sr=config.sample_rate, inv_preemphasize=config.preemphasize, k=config.preemphasis)

def save_checkpoint(sess, saver, checkpoint_path, global_step):
    saver.save(sess, checkpoint_path, global_step=global_step)

def model_train_mode(feeder, config, global_step, init=False):
    with tf.variable_scope('WaveNet_model', reuse=tf.AUTO_REUSE) as scope:
        model_name = 'WaveNet'
        model = create_model(model_name, config, init)
        #initialize model to train mode
        model.initialize(feeder.targets, feeder.local_condition_features, feeder.global_condition_features,
            feeder.input_lengths, x=feeder.inputs)
        model.add_loss(global_step)
        model.add_optimizer(global_step, config.post_train_mode, config.pre_train_mode, config.spk_train_mode) #spkclassifier
        stats = add_train_stats(model, config)
        return model, stats

def model_test_mode(feeder, config, global_step):
    with tf.variable_scope('WaveNet_model', reuse=tf.AUTO_REUSE) as scope:
        model_name = 'WaveNet'
        model = create_model(model_name, config)
        #initialize model to test mode
        model.initialize(feeder.eval_targets, feeder.eval_local_condition_features, feeder.eval_global_condition_features,
            feeder.eval_input_lengths)
        model.add_loss(global_step)
        return model

def main():

    # Instantiate Configs
    config = hparams

    # Directory Setting
    input_path = os.path.join(config.wavenet_input , 'map.txt')
    post_input_path = os.path.join(config.post_train_input , 'map.txt')     #post #yk: post train의 경우 input path설정
    log_dir = config.log_dir
    save_dir = os.path.join(log_dir, 'wave_pretrained')
    plot_dir = os.path.join(log_dir, 'plots')
    wav_dir = os.path.join(log_dir, 'wavs')
    eval_dir = os.path.join(log_dir, 'eval-dir')
    eval_plot_dir = os.path.join(eval_dir, 'plots')
    eval_wav_dir = os.path.join(eval_dir, 'wavs')
    tensorboard_dir = os.path.join(log_dir, 'wavenet_events')
    meta_folder = os.path.join(log_dir, 'metas')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(eval_plot_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(meta_folder, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'wavenet_model.ckpt')
    input_path = os.path.join(config.base_dir, input_path)
    post_input_path = os.path.join(config.base_dir, post_input_path)            #post #yk: post train의 경우 input path설정
    log.info('Checkpoint_path: {}'.format(checkpoint_path))
    if config.spk_train_mode == True:  # spk-train mode                     #post
        log.info('Loading spk-training data from: {}'.format(input_path))           #post
    elif config.post_train_mode == True:  # post-train mode                                                    #post
        log.info('Loading post-training data from: {}'.format(post_input_path)) #post
    else:   # train all vars                       #post
        log.info('Loading training data from: {}'.format(input_path))           #post
    log.info('Using model: {}'.format('WaveNet'))
    log.info(hparams_debug_string())

    #Start by setting a seed for repeatability
    tf.set_random_seed(config.wavenet_random_seed)

    #Set up data feeder
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as scope:
        if config.post_train_mode == True:  # post-train mode                                                    #post
            feeder = Feeder(coord, post_input_path, config.base_dir, config)        #post
        else :  # train all vars                       #post #yk : feeder는 post-train이냐 아니냐에따라 다른 data load
            feeder = Feeder(coord, input_path, config.base_dir, config)             #post
    #Instantiate Model Class (Graphing)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    model, stats = model_train_mode(feeder, config, global_step)
    eval_model = model_test_mode(feeder, config, global_step)  ##EVAL

    #Speaker Embeddings metadata
    if config.speakers_path is not None:
        speaker_embedding_meta = config.speakers_path

    else:
        speaker_embedding_meta = os.path.join(meta_folder, 'SpeakerEmbeddings.tsv')
        if not os.path.isfile(speaker_embedding_meta):
            with open(speaker_embedding_meta, 'w', encoding='utf-8') as f:
                for speaker in config.speakers:
                    f.write('{}\n'.format(speaker))

        speaker_embedding_meta = speaker_embedding_meta.replace(log_dir, '..')

    #book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    sh_saver = create_shadow_saver(model, global_step)

    if config.post_train_mode == True:
        log.info('Wavenet post training set to a maximum of {} steps'.format(config.post_train_steps))
        train_steps = config.post_train_steps
    elif config.pre_train_mode == True:
        log.info('Wavenet pre training set to a maximum of {} steps'.format(config.pre_train_steps))
        train_steps = config.pre_train_steps
    elif config.spk_train_mode == True:
        log.info('Wavenet spk training set to a maximum of {} steps'.format(config.spk_train_steps))
        train_steps = config.spk_train_steps
    else:  # train all vars                              #post  #yk : post train max step따로 받도록 >> train_step으로 통일 (뒤에 train loop에서 config.wavenet_train_steps이던 것을 train_steps로 바)
        log.info('Wavenet training set to a maximum of {} steps'.format(config.wavenet_train_steps))
        train_steps = config.wavenet_train_steps

    #Memory allocation on the memory
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = True
    run_init = False

    #Train
    with tf.Session(config=conf) as sess:
        try:
            summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            #saved model restoring
            if config.restore == True :
                # Restore saved model if the user requested it, default = True
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)

                    if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                        log.info('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
                        load_averaged_model(sess, sh_saver, checkpoint_state.model_checkpoint_path)
                    else:
                        log.info('No model to load at {}'.format(save_dir))
                        if config.wavenet_weight_normalization:
                            run_init = True

                except tf.errors.OutOfRangeError as e:
                    log.info('Cannot restore checkpoint: {}'.format(e))
            else:
                log.info('Starting new training!')
                if config.wavenet_weight_normalization:
                    run_init = True

            if run_init:
                log.info('\nApplying Weight normalization in fresh training. Applying data dependent initialization forward pass..')
                #Create init_model
                init_model, _ = model_train_mode(feeder, config, global_step, init=True)

            #initializing feeder
            feeder.start_threads(sess)

            if run_init:
                #Run one forward pass for model parameters initialization (make prediction on init_batch)
                _ = sess.run(init_model.tower_y_hat)
                log.info('Data dependent initialization done. Starting training!')

            #Training loop
            while not coord.should_stop() and step < train_steps:
                start_time = time.time()

                step, loss, vq_loss, vq_perplexity, reconst_loss, spk_loss, opt = sess.run([global_step, model.loss, model.vq_loss, model.vq_perplexity, model.reconst_loss, model.spk_loss, model.optimize])

                time_window.append(time.time() - start_time)
                loss_window.append(loss)

                message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}, vq_loss={:.5f}, vq_perplexity={:.5f}, reconst_loss={:.5f}, spk_loss={:.5f}]'.format(
                    step, time_window.average, loss, loss_window.average, vq_loss, vq_perplexity, reconst_loss, spk_loss)
                log.info(message)


                if np.isnan(loss) or loss > 10000:

                    log.info('Loss exploded to {:.5f} at step {}'.format(loss, step))
                    raise Exception('Loss exploded')

                if step % config.summary_interval == 0:
                    log.info('\nWriting summary at step {}'.format(step))
                    summary_writer.add_summary(sess.run(stats), step)

                if step % config.checkpoint_interval == 0 or step == train_steps:
                    save_log(sess, step, model, plot_dir, wav_dir, config=config)
                    save_checkpoint(sess, sh_saver, checkpoint_path, global_step)

                if step % config.eval_interval == 0:
                    log.info('\nEvaluating at step {}'.format(step))
                    eval_step(sess, step, eval_model, eval_plot_dir, eval_wav_dir, summary_writer=summary_writer , config=model._config) ##EVAL

                if config.gin_channels > 0 and (step % config.embedding_interval == 0 or step == train_steps): #or step == 1):
                    #Get current checkpoint state
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                    print("checkpoint_state : {}".format(checkpoint_state))

                    #Update Projector
                    log.info('\nSaving Model Speaker Embeddings visualization..')
                    add_embedding_stats(summary_writer, [model.embedding_table.name], [speaker_embedding_meta], checkpoint_state.model_checkpoint_path)
                    log.info('WaveNet Speaker embeddings have been updated on tensorboard!')

            log.info('Wavenet training complete after {} global steps'.format(train_steps))
            return save_dir

        except Exception as e:
            log.info('Exiting due to exception: {}'.format(e))
            traceback.print_exc()
            coord.request_stop(e)


if __name__ == "__main__":
    main()

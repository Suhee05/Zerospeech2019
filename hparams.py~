import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(


    #Synthesize
    synth_input_dir='/mnt/nas/03_ETC/zerospeech/databases',
    synth_language = 'surprise',
	wavenet_synth= 'synth_input/',
	synth_output_dir='synth_out/',
	synth_spk_id=None,
	is_training = True,
	synth_mode="all",# "embedding" or "wav" or "all"
	synth_idx=[0,10], # None or length 2 list. if not None, [start_idx, end_idx] will synthesize wavs_list[start_idx:end_idx]. Used only in synth_mode == "all"


	# For post training
	post_train_mode = False, #if True, post train mode / if False, train all variables
	post_train_input_dir = '/mnt/nas/03_ETC/zerospeech/databases/english/train/voice',
	post_train_input = './post_train_input/',
	post_train_steps = 900000,

	# For pre training
	pre_train_mode = False,
	pre_train_steps = 80000,

	# For Speaker classifier training
	spk_train_mode = True,
	spk_train_steps = 160000,

	#Directory
	base_dir = '',
	log_dir = 'logs',
	train_input_dir='/mnt/nas/03_ETC/zerospeech/databases/english/train/unit_voice',
	wavenet_input = 'train_input/', #tacotron_output/gta/map.txt

	restore = True, #Set this to False to do a fresh training
	summary_interval = 250, #Steps between running summary ops
	embedding_interval = 5000, #Steps between updating embeddings projection visualization
	checkpoint_interval = 2500, #Steps between writing checkpoints
	eval_interval = 5000, #Steps between eval on test data
	wavenet_train_steps = 500000, #total number of wavenet training steps

	tf_log_level = 1, #Tensorflow C++ log level
	wavenet_num_gpus = 1, #Determines the number of gpus in use for WaveNet training.
	split_on_cpu = True, #Determines whether to split data on CPU or on first GPU. This is automatically True when more than 1 GPU is used.


	###########################################################################################################################################

	# Preprocess
	num_mfccs = 39,
	num_freq = 1025,
	rescale = True,
	rescaling_max = 0.999,

	#train samples of lengths between 3sec and 14sec are more than enough to make a model capable of generating consistent speech.
	clip_mels_length = False, #For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)
	max_mel_frames = 900,  #Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.

	silence_threshold=2, #silence threshold used for sound trimming for wavenet preprocessing

	#MFCC for unit search

	n_fft = 400,
	hop_size = 160,
	win_size = 400,
	frame_shift_ms = None,
	magnitude_power = 2.,
	n_mfcc = 13,
	sample_rate = 16000,
	sample_size = 5120,
	num_mels = 80,
	sample_padding = False,

	#M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
	trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	trim_fft_size = 2048, #Trimming window size
	trim_hop_size = 512, #Trimmin hop length
	trim_top_db = 40, #Trimming db difference from reference db (smaller==harder trim.)

	#Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization = True, #Whether to normalize mel spectrograms to some predefined range (following below parameters)
	allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
	symmetric_mels = True, #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
	max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion,not too small for fast convergence)
	normalize_for_wavenet = True, #whether to rescale to [0, 1] for wavenet. (better audio quality)
	clip_for_wavenet = True, #whether to clip [-max, max] before training/synthesizing with wavenet (better audio quality)
	wavenet_pad_sides = 1, #Can be 1 or 2. 1 for pad right only, 2 for both sides padding.

	#Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
	preemphasize = True, #whether to apply filter
	preemphasis = 0.97, #filter coefficient.

	#Limits
	min_level_db = -100,
	ref_level_db = 20,
	fmin = 55,#55, #Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax = 7600, #To be increased/reduced depending on data.

	#Griffin Lim
	power = 1.5, #Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
	griffin_lim_iters = 60, #Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
	GL_on_GPU = True, #Whether to use G&L GPU version as part of tensorflow graph. (Usually much faster than CPU but slightly worse quality too).

	###########################################################################################################################################
	#VQVAE
	vqvae_batch_size = 16,
	vqvae_mfcc_dim = 39,
	vqvae_mfcc_frames =None,
	vqvae_num_training_updates = 50000,
	vqvae_enc_num_units = 768,
	vqvae_embedding_dim = 64,
	vqvae_num_embeddings = 256,
	vqvae_commitment_cost = 0.25,
	vqvae_vq_use_ema = False,
	vqvae_decay = 0.99,
	vqvae_learning_rate = 3e-4,
	vqvae_down_freq = 4, # when 100HZ -> 50HZ, down_freq is 2. when 100HZ -> 25HZ, it's 4.
	vqvae_dropout=0.23,

	###########################################################################################################################################
	#SpeakerClassifier
    clf_drop_out = 0.1,
    clf_lrelu_negative_slope = 0.01,

	###########################################################################################################################################
	#Wavenet

	#Model general type
	input_type="mulaw-quantize", #Raw has better quality but harder to train. mulaw-quantize is easier to train but has lower quality.
	quantize_channels=256,  # 65536 (16-bit) (raw) or 256 (8-bit) (mulaw or mulaw-quantize) // number of classes = 256 <=> mu = 255
	use_bias = True, #Whether to use bias in convolutional layers of the Wavenet
	legacy = True, #Whether to use legacy mode: Multiply all skip outputs but the first one with sqrt(0.5) (True for more early training stability, especially for large models)
	residual_legacy = True, #Whether to scale residual blocks outputs by a factor of sqrt(0.5) (True for input variance preservation early in training and better overall stability)

	#Model Losses parmeters
	log_scale_min=float(np.log(1e-14)), #Mixture of logistic distributions minimal log scale
	log_scale_min_gauss = float(np.log(1e-7)), #Gaussian distribution minimal allowed log scale
	#Loss type
	cdf_loss = False, #Whether to use CDF loss in Gaussian modeling. Advantages: non-negative loss term and more training stability. (Automatically True for MoL)

	#model parameters
	out_channels = 256, #This should be equal to quantize channels when input type is 'mulaw-quantize' else: num_distributions * 3 (prob, mean, log_scale).
	layers = 20, #Number of dilated convolutions (Default: Simplified Wavenet of Tacotron-2 paper)
	stacks = 2, #Number of dilated convolution stacks (Default: Simplified Wavenet of Tacotron-2 paper)
	residual_channels = 256, #Number of residual block input/output channels.
	gate_channels = 368, #split in 2 in gated convolutions
	skip_out_channels = 256, #Number of residual block skip convolution channels.
	kernel_size = 3, #The number of inputs to consider in dilated convolutions.

	#Upsampling parameters (local conditioning)
	cin_channels = 64, #Set this to -1 to disable local conditioning, else it must be equal to num_mels!!
	cin_first_conv_channels = 128,
	cin_first_conv_kernel_size = 3,
	upsample_type = 'NearestNeighbor', #Type of the upsampling deconvolution. Can be ('1D' or '2D', 'Resize', 'SubPixel' or simple 'NearestNeighbor').
	upsample_activation = 'Relu', #Activation function used during upsampling. Can be ('LeakyRelu', 'Relu' or None)
	upsample_scales = [16, 20], #prod(upsample_scales) should be equal to hop_size
	freq_axis_kernel_size = 3, #Only used for 2D upsampling types. This is the number of requency bands that are spanned at a time for each frame.
	leaky_alpha = 0.4, #slope of the negative portion of LeakyRelu (LeakyRelu: y=x if x>0 else y=alpha * x)
	NN_init = True, #Determines whether we want to initialize upsampling kernels/biases in a way to ensure upsample is initialize to Nearest neighbor upsampling. (Mostly for debug)
	NN_scaler = 0.3, #Determines the initial Nearest Neighbor upsample values scale. i.e: upscaled_input_values = input_values * NN_scaler (1. to disable)

	#global conditioning
	gin_channels = 16, #Set this to -1 to disable global conditioning, Only used for multi speaker dataset. It defines the depth of the embeddings (Recommended: 16)
	use_speaker_embedding = True, #whether to make a speaker embedding
	n_speakers = 112, #number of speakers (rows of the embedding)
	speakers_path = None, #Defines path to speakers metadata. Can be either in "speaker\tglobal_id" (with header) tsv format, or a single column tsv with speaker names. If None, use "speakers".
	speakers = ["S015","S020","S021","S023","S027","S031","S032","S033","S034","S035","S036","S037","S038","S039","S040","S041","S042","S043","S044","S045","S046","S047","S048","S049","S050","S051","S052","S053","S054","S055","S056","S058","S059","S060","S061","S063","S064","S065","S066","S067","S069","S070","S071","S072","S073","S074","S075","S076","S077","S078","S079","S080","S082","S083","S084","S085","S086","S087","S088","S090","S091","S092","S093","S094","S095","S096","S097","S098","S099","S100","S101","S102","S103","S104","S105","S106","S107","S109","S110","S111","S112","S113","S114","S115","S116","S117","S118","S119","S120","S121","S122","S123","S125","S126","S127","S128","S129","S131","S132","S133","V001","V002","S004","S010","S012","S014","S017","S018","S024","S025","S026","S028"],
									 #Must be consistent with speaker ids specified for global conditioning for correct visualization.

	###########################################################################################################################################

	#Wavenet Training
	wavenet_random_seed = 5339, # S=5, E=3, D=9 :)
	wavenet_data_random_state = 1234, #random state for train test split repeatability

	#performance parameters
	wavenet_swap_with_cpu = False, #Whether to use cpu as support to gpu for synthesis computation (while loop).(Not recommended: may cause major slowdowns! Only use when critical!)

	#train/test split ratios, mini-batches sizes
	wavenet_batch_size = 16, #batch size used to train wavenet.
	#During synthesis, there is no max_time_steps limitation so the model can sample much longer audio than 8k(or 13k) steps. (Audio can go up to 500k steps, equivalent to ~21sec on 24kHz)
	#Usually your GPU can handle ~2x wavenet_batch_size during synthesis for the same memory amount during training (because no gradients to keep and ops to register for backprop)
	wavenet_synthesis_batch_size = 1, #This ensure that wavenet synthesis goes up to 4x~8x faster when synthesizing multiple sentences. Watch out for OOM with long audios.
	wavenet_test_size = None, #% of data to keep as test data, if None, wavenet_test_batches must be not None
	wavenet_test_batches = 1, #number of test batches.

	#Learning rate schedule
	#wavenet_lr_schedule = 'exponential', #learning rate schedule. Can be ('exponential', 'noam')
	wavenet_lr_schedule ='piecewise',
	#wavenet_learning_rate = 1e-3, #wavenet initial learning rate
	wavenet_learning_rate = 4*1e-4,
	wavenet_warmup = float(4000), #Only used with 'noam' scheme. Defines the number of ascending learning rate steps.
	wavenet_decay_rate = 0.5, #Only used with 'exponential' scheme. Defines the decay rate.
	wavenet_decay_steps = 200000, #Only used with 'exponential' scheme. Defines the decay steps.
	wavenet_lr_boundaries =[400000, 600000, 800000],
	wavenet_lr_values=[4*1e-4, 2*1e-4, 1e-4, (0.5)*1e-4],

	#Optimization parameters
	wavenet_adam_beta1 = 0.9, #Adam beta1
	wavenet_adam_beta2 = 0.999, #Adam beta2
	wavenet_adam_epsilon = 1e-6, #Adam Epsilon

	#Regularization parameters
	wavenet_clip_gradients = True, #Whether the clip the gradients during wavenet training.
	wavenet_ema_decay = 0.9999, #decay rate of exponential moving average
	wavenet_weight_normalization = False, #Whether to Apply Saliman & Kingma Weight Normalization (reparametrization) technique. (Used in DeepVoice3, not critical here)
	wavenet_init_scale = 1., #Only relevent if weight_normalization=True. Defines the initial scale in data dependent initialization of parameters.
	wavenet_dropout = 0.05, #drop rate of wavenet layers
	wavenet_gradient_max_norm = 100.0, #Norm used to clip wavenet gradients
	wavenet_gradient_max_value = 5.0, #Value used to clip wavenet gradients

	#training samples length
	max_time_sec = None, #Max time of audio for training. If None, we use max_time_steps.
	max_time_steps = None, #Max time steps in audio used to train wavenet (decrease to save memory) (Recommend: 8000 on modest GPUs, 13000 on stronger ones)

	#Evaluation parameters
	wavenet_natural_eval = True, #Whether to use 100% natural eval (to evaluate autoregressivity performance) or with teacher forcing to evaluate overfit and model consistency.

	###########################################################################################################################################

	#Wavenet Debug
	wavenet_synth_debug = False, #Set True to use target as debug in WaveNet synthesis.
	wavenet_debug_wavs = ['wavenet_input/audio/audio-V002_4292252342.npy'], #Path to debug audios. Must be multiple of wavenet_num_gpus.
	wavenet_debug_mels = ['wavenet_input/mels/mel-V002_4292252342.npy'], #Path to corresponding mels. Must be of same length and order as wavenet_debug_wavs.

	)

def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)

# How to Run

### 1. Colab

Synthesis-ready Google Colab is uploaded here: https://colab.research.google.com/drive/1XNu8UltMOyW3_FPTizPgeb4UQW0Dad5J?authuser=1#scrollTo=GSj9SBSkONI2

Please be reminded that the CPU specs of Google Colab do not allow faster processing. 

#### 1.1 Installation

All packages and environments are already prepared in the Google Colab notebook above. 

#### 1.2 Run

Data preprocessing is already done for the colab version, so the respective code has been commented out.  Simply run the active cells step by step.

### 2. On your own system

#### 2.1. Installation  

```bash

#for stable installation, install these two packages as below
conda install -c conda-forge librosa
pip install -U dm-sonnet==1.23

# pip install r- requirements.txt
graphs==0.1.3
Keras==2.2.4
sonnet==0.1.6
tensorflow-gpu==1.12.0
tensorflow-probability==0.5.0
tensorflow-probability-gpu==0.4.0

```

#### 2.2. Revision to be made for the Sonnet source code

<https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py>

Please navigate to your /sonnet/python/modules/nets/vqvae.py installation location and make the following edits.

| line number | source                                                       | revised                                                      |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 98          | quantized = self.quantize(encoding_indices)                  | quantized, w = self.quantize(encoding_indices) </br> quantized_frozen = quantized |
| 111         | return {'quantize': quantized, 'loss': loss, 'perplexity': perplexity, 'encodings': encodings, 'encoding_indices': encoding_indices,} | return {'quantize': quantized, 'quantize_frozen':quantized_frozen, 'loss': loss, 'perplexity': perplexity, 'encodings': encodings, 'encoding_indices': encoding_indices, 'wmatrix':w,} |
| 121         | return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False) | return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False), w|

####2.3. Run

```bash
python preprocess.py --mode train
python train.py
python preprocess.py --mode synth
python synthesize.py
```


# Reference
- [1] Chorowski, J., Weiss, R. J., Bengio, S., & Oord, A. V. D. (2019). Unsupervised speech representation learning using WaveNet autoencoders. arXiv preprint arXiv:1901.08810.
- [2] Chou, Ju-chieh, Cheng-chieh Yeh, Hung-yi Lee, and Lin-shan Lee. “Multi-target Voice Conversion without Parallel Data by Adversarially Learning Disentangled Audio Representations.” arXiv preprint arXiv:1804.02812 (2018).
- [3] van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N. Senior, A.W., Kavukcuoglu, K. (2016). WaveNet: A generative model for raw audio. In SSW, p. 125.
- [4] van den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. In Advances in Neural Information Processing Systems (pp. 6306-6315).
- [5] https://github.com/Rayhane-mamah/Tacotron-2
- [6] https://github.com/deepmind/sonnet/blob/master/sonnet
- [7] https://github.com/jjery2243542/voice_conversion



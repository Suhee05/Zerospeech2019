from __future__ import print_function
from logger import log
import sonnet as snt
import tensorflow as tf


"""
[References]
1. Chou et al., Multi-target Voice Conversion without Parallel Data by Adversarially Learning Disentangled Audio Representations. (2018) (arXiv:1804.02812 [eess.AS])
2. https://github.com/jjery2243542/voice_conversion
"""

class SpeakerClassifier(snt.AbstractModule):

    def __init__(self, configs, name='speaker_classifier'):
        super(SpeakerClassifier, self).__init__(name=name)
        """Initialize Class
        Args:
            config: Processed configuration
        """
        self.config = configs
        self.dp = configs.clf_drop_out
        self.ns = configs.clf_lrelu_negative_slope
        self.name = "speaker_classifier"
        self.logger = log()        

    def __call__(self, input_placeholder):
        """Call Class. Initialize with Placeholders and Return Output Graph
        Args:
            input_placeholder: `3-D` tensor with shape `[batch_size, time_steps, c_channels]`.
            target: `2-D` tensor with shape `[batch_size, one_hot_size]`
        Returns:
            output: output graph
        Raises:
            TypeError
        """
        self.input = input_placeholder
        with tf.variable_scope(self.name):
            output = self._build(self.input)
        return output


    def _build(self, x):
        """Call Class. Initialize with Placeholders and Return Output Graph
        Args:
            x: `3-D` tensor with shape `[batch_size, time_steps, c_channels]`.
        Returns:
            speaker_classifier_output: `2-D` tensor with shape `[batch_size, one_hot_size]`
        Raises:
            TypeError
        """
        out1 = self.conv_block(x, 1, res=False)
        print("out1: {}".format(out1))
        out2 = self.conv_block(out1, 3, res=False)
        out3 = self.conv_block(out2, 5, res=False)
        out4 = self.conv_block(out3, 7, res=False)
        print("out4: {}".format(out4))
        out5 = self.softmax_layer(out4)  # 16,112,40
        print("out5: {}".format(out5))
        speaker_classifier_output = tf.reshape(out5, [tf.shape(out5)[0], -1])
        print("speaker_classifier_output: {}".format(speaker_classifier_output))
        return speaker_classifier_output


    def conv_layer(self, h, num_units, kernel_size, stride, name, padding=snt.SAME):
        h_i = snt.Conv1D(
        output_channels = num_units,
        kernel_shape = kernel_size, 
        stride = stride,
        padding = padding,
        data_format='NCW',
        name = name)(h)
        return h_i

    def conv_block(self, x, block_id, res=True):
        # padding
        out = self.pad_layer(x, 5)
        # convolution
        out = self.conv_layer(
            out,
            self.config.vqvae_embedding_dim, 
            kernel_size=5, 
            stride=1, 
            name="clf_layer{}".format(block_id))
        ## leaky RELU
        out = tf.nn.leaky_relu(out, alpha=self.ns, name="clf_lrelu{}".format(block_id))
        ## padding
        out = self.pad_layer(out, 5)
        ## convolution
        out = self.conv_layer(
            out,
            self.config.vqvae_embedding_dim, 
            kernel_size=5, 
            stride=1, 
            name="clf_layer{}".format(block_id+1))
        ## leaky RELU
        out = tf.nn.leaky_relu(out, alpha=self.ns, name="clf_lrelu{}".format(block_id+1))
        ## instance normalization
        out = tf.contrib.layers.instance_norm(out, data_format='NCHW')
        ## dropout
        out = tf.layers.dropout(out, rate=self.dp, name="clf_dropout{}".format(block_id))  
        # print("out : {}".format(out))
        if res:
            out = out + x
        return out

    def pad_layer(self, inp, kernel_size, is_2d=False):
        if type(kernel_size) == tuple:
            kernel_size = kernel_size[0]

        if not is_2d:
            if kernel_size % 2 == 0:
                pad = tf.constant([kernel_size//2, kernel_size//2 - 1])
            else:
                pad = tf.constant([[0,0], [0,0], [kernel_size//2, kernel_size//2]])   # BCT (pad on time dimension)
            
        else:
            if kernel_size % 2 == 0:
                pad = tf.constant([[kernel_size//2, kernel_size//2 - 1], [kernel_size//2, kernel_size//2 - 1]])
            else:
                pad = tf.constant([[kernel_size//2, kernel_size//2], [kernel_size//2, kernel_size//2]])
        
        # print("pad : {}".format(pad))
        # padding
        inp_padded = tf.pad(inp, 
                pad,
                mode='REFLECT')

        # print("input to be padded : {}".format(inp))
        # print("inp_padded : {}".format(inp_padded))
        return inp_padded

    def softmax_layer(self, x):
        out = self.conv_layer(
            x,
            self.config.n_speakers, 
            kernel_size=40, 
            stride=1, 
            name="softmax_layer",
            padding=snt.VALID)
        return out
 
from __future__ import print_function
import os
import subprocess
import tempfile
import glob
from logger import log
import matplotlib.pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf
import tarfile

from six.moves import cPickle
from six.moves import urllib
from six.moves import xrange


class VqEncoder(snt.AbstractModule):

    def __init__(self,configs, name='vqencoder'):
        super(VqEncoder, self).__init__(name=name)

        """Initialize Class
        Args:
            is_training: Boolean for indicating training/inference.
            config: Processed configuration
        """

        self.config = configs
        self.enc_num_units = configs.vqvae_enc_num_units #num_units = 768
        self.is_training = configs.is_training
        self.embedding_dim = configs.vqvae_embedding_dim
        self.vq_use_ema = configs.vqvae_vq_use_ema
        self.name = "vqencoder"
        self.logger = log()

    def __call__(self, input_placeholder):
        """Call Class. Initialize with Placeholders and Return Output Graph
        Args:
            input: `4-D` tensor with shape `[batch_size, width_size, height_size, channel_size]`.
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

        #usage
        # Conv1D.__init__(output_channels, kernel_shape, stride=1, rate=1,
        #     padding='SAME', use_bias=True, initializers=None, partitioners=None,
        #     regularizers=None, mask=None, data_format='NWC', custom_getter=None, name='conv_1d')

        ## conv3(768) ################################################################
        # conv1:
        # filterlen (kernel_shape) = 3
        # stride = 1
        # unit = 768 (768 filters)
        h = self.conv_layer(x,self.config.vqvae_enc_num_units,kernel_shape=3,stride=1, name="enc_layer1")
        h = tf.nn.relu(h)
        # print("enc_layer1 this is h {}".format(h))

        ## conv3(768) w/ residual connection ############################################
        # with residual
        # conv2:
        # filterlen (kernel_shape) = 3
        # stride=1
        # unit=768
        h = self.conv_with_residual(h, self.config.vqvae_enc_num_units,kernel_shape=3,stride=1,name="enc_layer2")
        h = tf.nn.relu(h)
        # print("enc_layer2_with_residual this is h {}".format(h))

        ## StridedConv4(768) ###########################################################
        # length reducing layer:
        # filterlen (kernel_shape) = 4
        # stride=2
        # unit=768
        h = self.conv_layer(h, self.config.vqvae_enc_num_units,kernel_shape=4,stride=2,name="enc_layer3")
        h = tf.nn.relu(h)
        # print("enc_layer3 REDUCTION1 to 50hz - this is h {}".format(h))

        ## StridedConv4(768) ###########################################################
        # length reducing layer:
        # filterlen (kernel_shape) = 4
        # stride=2
        # unit=768
        h = self.conv_layer(h, self.config.vqvae_enc_num_units,kernel_shape=4,stride=2,name="enc_layer4")
        h = tf.nn.relu(h)
        # print("enc_layer4 REDUCTION2 to 25hz - this is h {}".format(h))




        ## conv3(768) w/ residual connection ###########################################
        # conv3:
        # filterlen (kernel_shape) = 3
        # stride=1
        # unit=768 (768 filters)

        h = self.conv_with_residual(h, self.config.vqvae_enc_num_units,kernel_shape=3,stride=1,name="enc_layer4")
        h = tf.nn.relu(h)
        # print("enc_layer5_w_residual this is h {}".format(h))

        ## conv3(768) w/ residual connection ###########################################
        # conv4:
        # filterlen (kernel_shape) = 3
        # stride = 1
        # unit=768 (768 filters)
        h = self.conv_with_residual(h, self.config.vqvae_enc_num_units,kernel_shape=3,stride=1,name="enc_layer5")
        h = tf.nn.relu(h)
        # print("enc_layer6_w_residual this is h {}".format(h))

        ## Relu(768) w/ residual connection ###########################################
        ### 1
        h = self.relu_with_residual(h)
        # print("enc_layer7_relu {}".format(h))
        ### 2
        h = self.relu_with_residual(h)
        # print("enc_layer8_relu this is h {}".format(h))
        ### 3
        h = self.relu_with_residual(h)
        # print("enc_layer9_relu this is h {}".format(h))
        ### 4
        h = self.relu_with_residual(h)
        # print("enc_layer10_relu this is h {}".format(h))


        #Vector Quantize result
        ## transform dim to vq embedding (linear)
        z = tf.layers.dense(h,self.config.vqvae_embedding_dim)
        # print("pre-vq-dim-correction {} ".format(z))

        ## define vq_vae func
        vq_vae = self.def_vq(self.config.vqvae_vq_use_ema)

        ## do vq
        vq_output_train = vq_vae(z, is_training=self.config.is_training)


        return vq_output_train


    def conv_layer(self,h,num_units,kernel_shape,stride,name):

        h_i = snt.Conv1D(
        output_channels = num_units,
        kernel_shape = kernel_shape, #
        stride = stride,
        data_format='NWC',
        name = name)(h)

        return h_i

    #define conv layer with residual connection
    def conv_with_residual(self,h,num_units,kernel_shape,stride,name):

        h_i = snt.Conv1D(
        output_channels = num_units,
        kernel_shape = kernel_shape, #
        stride = stride,
        data_format='NWC',
        name = name)(h)

        h_i = tf.nn.relu(h_i)
        # print("this conv_with_residual is h_i {}".format(h_i))
        # print("this conv_with_residual is residual {}".format(h))
        h += h_i

        return h


    def relu_with_residual(self,h):

        h_i = tf.nn.relu(h)
        h += h_i
        return h

    def prep_vq(self,embedding_dim):

        pre_vq_conv1 = snt.Conv1D(output_channels=self.config.vqvae_embedding_dim,
        kernel_shape=1,
        stride=1,
        name="linear_to_vq_dim")

        return pre_vq_conv1

    def def_vq(self,vq_use_ema):

        if vq_use_ema:
          vq_vae = snt.nets.VectorQuantizerEMA(
              embedding_dim=self.config.vqvae_embedding_dim,
              num_embeddings=self.config.vqvae_num_embeddings,
              commitment_cost=self.config.vqvae_commitment_cost,
              decay=self.config.vqvae_decay)
        else:
          vq_vae = snt.nets.VectorQuantizer(
              embedding_dim=self.config.vqvae_embedding_dim,
              num_embeddings=self.config.vqvae_num_embeddings,
              commitment_cost=self.config.vqvae_commitment_cost)

        return vq_vae


################################################## 이 부분??

    def add_loss(self,VqEncoder_out):
        """ Calculate Loss
        Args:
        Returns:
        Raises:
        """
        with tf.variable_scope('loss') as scope:
            self.loss = VqEncoder_out["loss"]

    def add_opt(self, global_step):
        """ Load Optimizer and Calculate Optimization
        Args:
        Returns:
        Raises:
        """
        with tf.variable_scope('optimizer') as scope:
            optimizer = tf.train.AdamOptimizer(self.config.vqvae_learning_rate)
            self.opt = optimizer.minimize(self.loss, global_step=global_step)

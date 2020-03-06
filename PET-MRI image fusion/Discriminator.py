import tensorflow as tf
import numpy as npimport

import tensorflow as tf

from tensorflow.python import pywrap_tensorflow
import numpy as np

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.1


class Discriminator1(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.variable_scope(scope_name):
			self.weight_vars.append(self._create_variables(1, 16, 3, scope = 'conv1'))
			self.weight_vars.append(self._create_variables(16, 32, 3, scope = 'conv2'))
			self.weight_vars.append(self._create_variables(32, 64, 3, scope = 'conv3'))
			#self.weight_vars.append(self._create_variables(64, 96, 3, scope = 'conv4'))
			#self.weight_vars.append(self._create_variables(96, 128, 3, scope = 'conv5'))
		# self.weight_vars.append(self._create_variables(128, 256, 3, scope = 'conv6'))
		# self.weight_vars.append(self._create_variables(12, 1, 3, scope = 'conv6'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def discrim(self, img, reuse):
		conv_num = len(self.weight_vars)
		if len(img.shape) != 4:
			img = tf.expand_dims(img, -1)
		# if len(c.shape) != 4:
		# 	c = tf.expand_dims(c, -1)
		# out = tf.concat([c, img], axis = 3)
		out = img
		for i in range(conv_num):
			kernel, bias = self.weight_vars[i]
			if i == 0:
				out = conv2d_1(out, kernel, bias, [1, 2, 2, 1], use_relu = True, use_BN = False,
				               Scope = self.scope + '/b' + str(i), Reuse = reuse)
			# elif i == conv_num - 1:
			# 	out = tf.nn.conv2d(out, kernel, [1, 1, 1, 1], padding = 'VALID')
			# 	out = tf.nn.bias_add(out, bias)
			# 	out = tf.nn.tanh(out)
			# 	out = out / 2 + 0.5
			else:
				out = conv2d_1(out, kernel, bias, [1, 2, 2, 1], use_relu = True, use_BN = True,
				               Scope = self.scope + '/b' + str(i), Reuse = reuse)
		out = tf.reshape(out, [-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
		with tf.variable_scope(self.scope):
			with tf.variable_scope('flatten1'):
				# 	out = tf.layers.dense(out, 512, activation = tf.nn.relu, use_bias = True, trainable = True,
				# 	# 	                      reuse = reuse)
				# 	out = tf.layers.batch_normalization(out, training = True, reuse = reuse)
				# 	# with tf.variable_scope('flatten2'):
				out = tf.layers.dense(out, 1, activation = tf.nn.tanh, use_bias = True, trainable = True,
				                      reuse = reuse)
		out = out / 2 + 0.5
		return out


def conv2d_1(x, kernel, bias, strides, use_relu = True, use_BN = True, Scope = None, Reuse = None):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides, padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if use_BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = True, reuse = Reuse)
	if use_relu:
		out = tf.nn.relu(out)
	return out


class Discriminator2(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.variable_scope(scope_name):
			self.weight_vars.append(self._create_variables(1, 16, 3, scope = 'conv1'))
			self.weight_vars.append(self._create_variables(16, 32, 3, scope = 'conv2'))
			self.weight_vars.append(self._create_variables(32, 64, 3, scope = 'conv3'))
		# self.weight_vars.append(self._create_variables(96, 128, 3, scope = 'conv4'))
		# self.weight_vars.append(self._create_variables(64, 96, 3, scope = 'conv4'))
		# self.weight_vars.append(self._create_variables(96, 128, 3, scope = 'conv5'))
		# self.weight_vars.append(self._create_variables(128, 256, 3, scope = 'conv4'))
		# self.weight_vars.append(self._create_variables(256, 512, 3, scope = 'conv5'))
		# self.weight_vars.append(self._create_variables(12, 1, 3, scope = 'conv6'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def discrim(self, img, reuse):
		conv_num = len(self.weight_vars)
		if len(img.shape) != 4:
			img = tf.expand_dims(img, -1)
		# if len(c.shape) != 4:
		# 	c = tf.expand_dims(c, -1)
		# out = tf.concat([c, img], axis = 3)
		out = img
		for i in range(conv_num):
			kernel, bias = self.weight_vars[i]
			if i == 0:
				out = conv2d_2(out, kernel, bias, [1, 2, 2, 1], use_relu = True, use_BN = False,
				               Scope = self.scope + '/b' + str(i), Reuse = reuse)
			# elif i == conv_num - 1:
			# 	out = tf.nn.conv2d(out, kernel, [1, 1, 1, 1], padding = 'VALID')
			# 	out = tf.nn.bias_add(out, bias)
			# 	out = tf.nn.tanh(out)
			# 	out = out / 2 + 0.5
			else:
				out = conv2d_2(out, kernel, bias, [1, 2, 2, 1], use_relu = True, use_BN = True,
				               Scope = self.scope + '/b' + str(i), Reuse = reuse)
		out = tf.reshape(out, [-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
		with tf.variable_scope(self.scope):
			with tf.variable_scope('flatten1'):
				# 	out = tf.layers.dense(out, 512, activation = tf.nn.relu, use_bias = True, trainable = True,
				# 	# 	                      reuse = reuse)
				# 	out = tf.layers.batch_normalization(out, training = True, reuse = reuse)
				# 	# with tf.variable_scope('flatten2'):
				out = tf.layers.dense(out, 1, activation = tf.nn.tanh, use_bias = True, trainable = True,
				                      reuse = reuse)
		out = out / 2 + 0.5
		return out


def conv2d_2(x, kernel, bias, strides, use_relu = True, use_BN = True, Scope = None, Reuse = None):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides, padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if use_BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = True, reuse = Reuse)
	if use_relu:
		out = tf.nn.relu(out)
	return out

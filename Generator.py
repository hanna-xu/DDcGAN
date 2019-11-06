import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.05


class Generator(object):

	def __init__(self, sco):
		self.encoder = Encoder(sco)
		self.decoder = Decoder(sco)

	def transform(self, vis, ir):
		ir1 = up_sample(ir, scale_factor = 2)
		IR = up_sample(ir1, scale_factor = 2)
		img = tf.concat([vis, IR], 3)
		code2, code4, code5 = self.encoder.encode(img)
		generated_img = self.decoder.decode(code2, code4, code5)
		return generated_img


class Encoder(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []
		with tf.variable_scope(self.scope):
			with tf.variable_scope('encoder'):
				self.weight_vars.append(self._create_variables(2, 48, 3, scope = 'conv1_1'))
				self.weight_vars.append(self._create_variables(48, 96, 3, scope = 'conv1_2'))
				self.weight_vars.append(self._create_variables(96, 144, 3, scope = 'conv1_3'))
				self.weight_vars.append(self._create_variables(144, 192, 3, scope = 'conv1_4'))
				self.weight_vars.append(self._create_variables(192, 240, 3, scope = 'conv1_5'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def encode(self, image):
		out0 = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]

			if i == 0:
				out1 = conv2d(out0, kernel, bias, use_relu = True, Scope = self.scope + '/encoder/b' + str(i))
			if i == 1:
				out2 = conv2d(out1, kernel, bias, use_relu = True, Scope = self.scope + '/encoder/b' + str(i))
			if i == 2:
				out3 = conv2d(out2, kernel, bias, use_relu = True, Scope = self.scope + '/encoder/b' + str(i),
				              strides = [1, 2, 2, 1])
			if i == 3:
				out4 = conv2d(out3, kernel, bias, use_relu = True, Scope = self.scope + '/encoder/b' + str(i))
			if i == 4:
				out5 = conv2d(out4, kernel, bias, use_relu = True, Scope = self.scope + '/encoder/b' + str(i),
				              strides = [1, 2, 2, 1])
		return (out2, out4, out5)


class Decoder(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.name_scope(scope_name):
			with tf.variable_scope('decoder'):
				self.weight_vars.append(self._create_variables(240, 240, 3, scope = 'conv2_1'))
				self.weight_vars.append(self._create_variables(240 + 192, 128, 3, scope = 'conv2_2'))
				self.weight_vars.append(self._create_variables(128, 64, 3, scope = 'conv2_3'))
				self.weight_vars.append(self._create_variables(64 + 96, 32, 3, scope = 'conv2_4'))
				self.weight_vars.append(self._create_variables(32, 1, 3, scope = 'conv2_5'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		with tf.variable_scope(scope):
			shape = [kernel_size, kernel_size, input_filters, output_filters]
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def decode(self, code2, code4, code5):
		final_layer_idx = len(self.weight_vars) - 1
		out0 = code5
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			if i == 0:
				out1 = conv2d(out0, kernel, bias, use_relu = True, Scope = self.scope + '/decoder/b' + str(i),
				              BN = False)
				out1 = up_sample(out1, scale_factor = 2)
			if i == 1:
				out2 = conv2d(tf.concat([out1, code4], 3), kernel, bias, use_relu = True, BN = True,
				              Scope = self.scope + '/decoder/b' + str(i))
			if i == 2:
				out3 = conv2d(out2, kernel, bias, use_relu = True, BN = True,
				              Scope = self.scope + '/decoder/b' + str(i))
				out3 = up_sample(out3, scale_factor = 2)
			if i == 3:
				out4 = conv2d(tf.concat([out3, code2], 3), kernel, bias, use_relu = True, BN = True,
				              Scope = self.scope + '/decoder/b' + str(i))
			if i == final_layer_idx:
				out = conv2d(out4, kernel, bias, use_relu = False, Scope = self.scope + '/decoder/b' + str(i),
				             BN = False)
				out = tf.nn.tanh(out) / 2 + 0.5
		return out


def conv2d(x, kernel, bias, use_relu = True, Scope = None, BN = True, strides = [1, 1, 1, 1]):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides, padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = True)
	if use_relu:
		out = tf.nn.relu(out)

	return out


def up_sample(x, scale_factor = 2):
	_, h, w, _ = x.get_shape().as_list()
	new_size = [h * scale_factor, w * scale_factor]
	return tf.image.resize_nearest_neighbor(x, size = new_size)

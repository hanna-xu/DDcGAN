import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave
from datetime import datetime
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
import time


def generate(ir_path, vis_path, model_path, output_path = None):
	ir_img = imread(ir_path) / 255.0
	vis_img = imread(vis_path) / 255.0
	ir_dimension = list(ir_img.shape)
	vis_dimension = list(vis_img.shape)
	ir_dimension.insert(0, 1)
	ir_dimension.append(1)
	vis_dimension.insert(0, 1)
	vis_dimension.append(1)
	ir_img = ir_img.reshape(ir_dimension)
	vis_img = vis_img.reshape(vis_dimension)

	with tf.Graph().as_default(), tf.Session() as sess:
		SOURCE_VIS = tf.placeholder(tf.float32, shape = vis_dimension, name = 'SOURCE_VIS')
		SOURCE_ir = tf.placeholder(tf.float32, shape = ir_dimension, name = 'SOURCE_ir')

		G = Generator('Generator')
		# D1 = Discriminator1('Discriminator1')
		# D2 = Discriminator2('Discriminator2')

		output_image = G.transform(vis = SOURCE_VIS, ir = SOURCE_ir)
		# grad_VIS = grad(SOURCE_VIS)
		# grad_output_image = grad(output_image)
		# g0 = tf.nn.avg_pool(output_image, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
		# output_image_ds = tf.nn.avg_pool(g0, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

		saver = tf.train.Saver()
		saver.restore(sess, model_path)
		output = sess.run(output_image, feed_dict = {SOURCE_VIS: vis_img, SOURCE_ir: ir_img})
		output = output[0, :, :, 0]
		print('output shape:', output.shape)
		imsave(output_path + 'fused_result.bmp', output)


def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g
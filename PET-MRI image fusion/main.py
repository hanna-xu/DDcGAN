from __future__ import print_function

import time

# from utils import list_images
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
from generate import generate
import scipy.ndimage
import tensorflow as tf
from scipy.misc import imread, imsave

BATCH_SIZE = 26
EPOCHES = 1
LOGGING = 20
MODEL_SAVE_PATH = './models/'
output_path='./fused_imgs/'
IS_TRAINING = False

f = h5py.File('Dataset_same_resolution.h5', 'r')
# for key in f.keys():
#   print(f[key].name)
sources = f['data'][:]
sources = np.transpose(sources, (0, 3, 2, 1))
print("source shape:", sources.shape)

# for i in range(int(sources.shape[0])):
# 	ir_ds = scipy.ndimage.zoom(sources[i, :, :, 1], 0.25)
# 	ir_ds_us = scipy.ndimage.zoom(ir_ds, 4, order = 3)
# 	sources[i, :, :, 1] = ir_ds_us
#
# if not os.path.exists('Dataset3_ds_us.h5'):
# 	with h5py.File('Dataset3_ds_us.h5') as f2:
# 		f2['data'] = sources

def main():
	if IS_TRAINING:
		print(('\nBegin to train the network ...\n'))
		train(sources, MODEL_SAVE_PATH, EPOCHES, BATCH_SIZE, logging_period = LOGGING)
	else:
		print('\nBegin to generate pictures ...\n')
		path = './test_imgs/'

		# for root, dirs, files in os.walk(path):
		# 	test_num = len(files)

		for k in range(12):
			model_num = 20 * (k + 33)
			t = []
			# Except = [5, 6, 13, 17, 19]
			for i in range(20):
				index = i+1
				savepath = './fused_imgs/' + str(index) + '/'
				# if index not in Except:
				ir_path = path + 'IR' + str(index) + '.bmp'
				vis_path = path + 'VIS' + str(index) + '.bmp'
				# generate(ir_path, vis_path, MODEL_SAVE_PATH + str(model_num) + '/' + str(model_num) + '.ckpt', index,
				#          model_num = model_num, output_path = savepath)
				model_path = './models/' + str(model_num) + '/' + str(model_num) + '.ckpt'
				output, Time = generate(ir_path, vis_path, model_path, model_num = model_num, output_path = savepath)
				print("model_num:%s, pic_num:%s" % (model_num, index))
				t.append(Time)

				# fig = plt.figure()
				# fig1 = fig.add_subplot(111)
				# fig1.imshow(output, cmap = 'gray')
				# plt.show()
				save_path=output_path +str(index)
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				imsave(save_path+'/' + str(model_num) + '.bmp', output)
		print("Time: mean:%s, std: %s" % (np.mean(t), np.std(t)))


if __name__ == '__main__':
	main()

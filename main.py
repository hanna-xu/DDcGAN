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

BATCH_SIZE = 24
EPOCHES = 1
LOGGING = 50
MODEL_SAVE_PATH = './models/'

f = h5py.File('Training_Dataset.h5', 'r')
# for key in f.keys():
#   print(f[key].name)
sources = f['data'][:]
# sources = np.transpose(a, (0, 3, 2, 1))

IS_TRAINING = False


def main():
	if IS_TRAINING:
		print(('\nBegin to train the network ...\n'))
		train(sources, MODEL_SAVE_PATH, EPOCHES, BATCH_SIZE, logging_period = LOGGING)
	else:
		print('\nBegin to generate pictures ...\n')
		test_path = './test_imgs/'
		savepath = './results/'
		model_path = MODEL_SAVE_PATH + 'model/model.ckpt'

		ir_path = test_path + 'IR13_ds.bmp'
		vis_path = test_path + 'VIS13.bmp'

		begin = time.time()
		generate(ir_path, vis_path, model_path, output_path = savepath)
		end = time.time()
		print("time:%s" % (end - begin))


if __name__ == '__main__':
	main()

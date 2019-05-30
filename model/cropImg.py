# crop preprocessed images, save bbox and landmark

import os
import cv2
import numpy as np

crop_train_file = '/home/zhangyuqi/NewDisk/train_label.txt'
crop_test_file = '/home/zhangyuqi/NewDisk/test_label.txt'
crop_valid_file = '/home/zhangyuqi/NewDisk/valid_label.txt'

def crop_img(img_path, label_path, export_img, export_label):
	with open(label_path, 'r') as f:
		while True:
			line = f.readline()
			imgPath = line.split()[0]
			point = line.split()[1:]
			img = cv2.imread(imgPath)
			n_p = 68


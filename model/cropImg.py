# crop preprocessed images, save bbox and landmark

import os
import cv2
import numpy as np
from showImage import show_landmarks

crop_train_file = '/home/zhangyuqi/NewDisk/train_label.txt'
crop_test_file = '/home/zhangyuqi/NewDisk/test_label.txt'
crop_valid_file = '/home/zhangyuqi/NewDisk/valid_label.txt'

def crop_img(img_origin, landmarks):
	'''
	crop image according to the landmarks,
	:param img_origin: original image before crop
	:param landmarks:
	:return: cropped image
	'''
	img = img_origin.copy()
	h, w, c = img.shape
	lk = np.array(landmarks).astype(float)
	lk = lk.reshape(-1, 2)
	lk1 = lk[:, 0]
	lk2 = lk[:, 1]
	w_start = lk1.min()
	w_end = lk1.max()
	h_start = lk2.min()
	h_end = lk2.max()
	w_crop = w_end - w_start
	h_crop = h_end - h_start

	w_start = int(w_start - w_crop / 4)
	w_end = int(w_end + w_crop / 4)
	h_start = int(h_start - h_crop / 4)
	h_end = int(h_end + h_crop / 4)

	if w_start < 0:
		w_start = 0
	if h_start < 0:
		h_start = 0
	if w_end > w:
		w_end = w
	if h_end > h:
		h_end = h

	# calculate new landmarks

	lk[:, 0] = lk[:, 0] - w_start
	lk[:, 1] = lk[:, 1] - h_start
	out_lk = []
	for i in range(lk.shape[0]):
		out_lk.append(lk[i][0])
		out_lk.append(lk[i][1])

	img = img[h_start: h_end+1, w_start: w_end+1]

	return img, out_lk

def main():
	f = open(crop_valid_file, 'r')
	annotion = f.readline()
	img_path = annotion.split()[0]
	landmarks = annotion.split()[1:]
	img = cv2.imread(img_path)
	img, landmarks = crop_img(img, landmarks)
	show_landmarks(img, landmarks)
	f.close()

if __name__ == '__main__':
    main()




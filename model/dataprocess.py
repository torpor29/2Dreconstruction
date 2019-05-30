
import numpy as np
import cv2
import random
import  sys
sys.path.append('/home/zhangyuqi/clone/facedetect')
import os
import glob
import shutil

# ---------------------------------calculate the mean and std------------------------------------
def main():

	# train_txt_path = '/home/zhangyuqi/clone/facedetect/1_level_1/Data/l1_train_label.txt'
	# img_path_root = '/home/zhangyuqi/clone/facedetect/1_level_1/Data/train/'
	train_txt_path = '/home/zhangyuqi/NewDisk/train_label.txt'
	img_path_root = '/home/zhangyuqi/NewDisk/train/'

	CNum = 2000

	img_h, img_w = 48, 48
	imgs = np.zeros([img_w, img_h, 3, 1])
	means, stdevs = [], []
	with open(train_txt_path, 'r') as f:
		lines = f.readlines()
		random.shuffle(lines)

		for i in range(CNum):
			img_path = lines[i].rstrip().split(' ')[0]
			img = cv2.imread(img_path)
			img = cv2.resize(img, (img_h, img_w))

			img = img[:, :, :, np.newaxis]
			imgs = np.concatenate((imgs, img), axis=3)

		imgs = imgs.astype(np.float32)/255

	for i in range(3):
		pixels = imgs[:, :, i, :].ravel()
		means.append(np.mean(pixels))
		stdevs.append(np.std(pixels))

	means.reverse()
	stdevs.reverse()

	print("normMean'+{}".format(means))
	print("normStd={}".format(stdevs))
	print("transforms.Normalize(normMean={},normStd={}".format(means, stdevs))

def makedir(new_dir):
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)

def data_split():
	dataset_dir = '/home/zhangyuqi/clone/facedetect/1_level_1/Data/trainraw/'
	datalabel_path = '/home/zhangyuqi/clone/facedetect/1_level_1/Data/l1_train_label_raw.txt'

	valid_dir = '/home/zhangyuqi/clone/facedetect/1_level_1/Data/valid/'
	train_dir = '/home/zhangyuqi/clone/facedetect/1_level_1/Data/train/'
	valid_label = '/home/zhangyuqi/clone/facedetect/1_level_1/Data/l1_valid_label.txt'
	train_label = '/home/zhangyuqi/clone/facedetect/1_level_1/Data/l1_train_label.txt'
	train_per = 0.9
	valid_per = 0.1

	makedir(valid_dir)
	makedir(train_dir)
	# makedir(valid_label)
	# makedir(train_label)
	# tf = open(train_dir, 'w')
	# vf = open(valid_dir, 'w')
	tf_label = open(train_label, 'w')
	vf_label = open(valid_label, 'w')
	# df = open(dataset_dir, 'r')
	df_label = open(datalabel_path, 'r')
	lines = df_label.readlines()
	random.shuffle(lines)
	for i in range(int(len(lines)*train_per)):
		img_path = dataset_dir + lines[i].rstrip().split()[0]
		img = cv2.imread(img_path)
		lines[i] = train_dir + lines[i]
		tf_label.writelines(lines[i])
		img_path_train = lines[i].rstrip().split()[0]
		cv2.imwrite(img_path_train, img)

	for j in range(int(len(lines)*valid_per)):
		img_path = dataset_dir + lines[j+int(len(lines)*train_per)].rstrip().split()[0]
		img = cv2.imread(img_path)
		lines[j + int(len(lines) * train_per)] = valid_dir + lines[j + int(len(lines) * train_per)]
		vf_label.writelines(lines[j+int(len(lines)*train_per)])
		img_path_valid = lines[j+int(len(lines)*train_per)].rstrip().split()[0]
		cv2.imwrite(img_path_valid, img)

	# tf.close()
	# vf.close()
	tf_label.close()
	vf_label.close()
	df_label.close()

if __name__ == '__main__':
	main()
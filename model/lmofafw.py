# rewrite the landmark to label.txt file(afw, helen, ibug, ifpw)

import math
import os
import random
import cv2
from showImage import show_landmarks
from cropImg import crop_img
# rewrite the landmark to label.txt file (for original images)
afw_path = '/home/zhangyuqi/NewDisk/afw'
helen_test_path = '/home/zhangyuqi/NewDisk/helen/testset'
helen_train_path = '/home/zhangyuqi/NewDisk/helen/trainset'
ibug_path = '/home/zhangyuqi/NewDisk/ibug'
ifpw_test_path = '/home/zhangyuqi/NewDisk/ifpw/testset'
ifpw_train_path = '/home/zhangyuqi/NewDisk/ifpw/trainset'
afw_file = '/home/zhangyuqi/projects/model/afw_raw_label.txt'
helen_test_file = '/home/zhangyuqi/projects/model/helen_test_raw_label.txt'
helen_train_file = '/home/zhangyuqi/projects/model/helen_train_raw_label.txt'
ibug_file = '/home/zhangyuqi/projects/model/ibug_raw_label.txt'
ifpw_test_file = '/home/zhangyuqi/projects/model/ifpw_test_raw_label.txt'
ifpw_train_file = '/home/zhangyuqi/projects/model/ifpw_train_raw_label.txt'
# ----------------------------------------------------------------------------
perTrain = 0.7  # percentage of train set
perValid = 0.1
perTest = 0.2
img_w = 160
img_h = 160
# rewrite the landmark to label.txt file (for preprocessed images)
input_path = ['/home/zhangyuqi/NewDisk/300W/01_Indoor',
              '/home/zhangyuqi/NewDisk/300W/02_Outdoor',
              '/home/zhangyuqi/NewDisk/300W_operate/01_Indoor/lightness_0',
              '/home/zhangyuqi/NewDisk/300W_operate/01_Indoor/lightness_1',
              '/home/zhangyuqi/NewDisk/300W_operate/01_Indoor/rotate15',
              '/home/zhangyuqi/NewDisk/300W_operate/01_Indoor/rotate30',
              '/home/zhangyuqi/NewDisk/300W_operate/01_Indoor/transpose',
              '/home/zhangyuqi/NewDisk/300W_operate/02_Outdoor/lightness_0',
              '/home/zhangyuqi/NewDisk/300W_operate/02_Outdoor/lightness_1',
              '/home/zhangyuqi/NewDisk/300W_operate/02_Outdoor/rotate15',
              '/home/zhangyuqi/NewDisk/300W_operate/02_Outdoor/rotate30',
              '/home/zhangyuqi/NewDisk/300W_operate/02_Outdoor/transpose',
              '/home/zhangyuqi/NewDisk/afw',
              '/home/zhangyuqi/NewDisk/afw_operate/lightness_0',
              '/home/zhangyuqi/NewDisk/afw_operate/lightness_1',
              '/home/zhangyuqi/NewDisk/afw_operate/rotate15',
              '/home/zhangyuqi/NewDisk/afw_operate/rotate30',
              '/home/zhangyuqi/NewDisk/afw_operate/transpose',
              '/home/zhangyuqi/NewDisk/helen/testset',
              '/home/zhangyuqi/NewDisk/helen/trainset',
              '/home/zhangyuqi/NewDisk/helen_operate/testset/lightness_0',
              '/home/zhangyuqi/NewDisk/helen_operate/testset/lightness_1',
              '/home/zhangyuqi/NewDisk/helen_operate/testset/rotate15',
              '/home/zhangyuqi/NewDisk/helen_operate/testset/rotate30',
              '/home/zhangyuqi/NewDisk/helen_operate/testset/transpose',
              '/home/zhangyuqi/NewDisk/helen_operate/trainset/lightness_0',
              '/home/zhangyuqi/NewDisk/helen_operate/trainset/lightness_1',
              '/home/zhangyuqi/NewDisk/helen_operate/trainset/rotate15',
              '/home/zhangyuqi/NewDisk/helen_operate/trainset/rotate30',
              '/home/zhangyuqi/NewDisk/helen_operate/trainset/transpose',
              '/home/zhangyuqi/NewDisk/ibug',
              '/home/zhangyuqi/NewDisk/ibug_operate/lightness_0',
              '/home/zhangyuqi/NewDisk/ibug_operate/lightness_1',
              '/home/zhangyuqi/NewDisk/ibug_operate/rotate15',
              '/home/zhangyuqi/NewDisk/ibug_operate/rotate30',
              '/home/zhangyuqi/NewDisk/ibug_operate/transpose',
              '/home/zhangyuqi/NewDisk/ifpw/testset',
              '/home/zhangyuqi/NewDisk/ifpw/trainset',
              '/home/zhangyuqi/NewDisk/ifpw_operate/testset/lightness_0',
              '/home/zhangyuqi/NewDisk/ifpw_operate/testset/lightness_1',
              '/home/zhangyuqi/NewDisk/ifpw_operate/testset/rotate15',
              '/home/zhangyuqi/NewDisk/ifpw_operate/testset/rotate30',
              '/home/zhangyuqi/NewDisk/ifpw_operate/testset/transpose',
              '/home/zhangyuqi/NewDisk/ifpw_operate/trainset/lightness_0',
              '/home/zhangyuqi/NewDisk/ifpw_operate/trainset/lightness_1',
              '/home/zhangyuqi/NewDisk/ifpw_operate/trainset/rotate15',
              '/home/zhangyuqi/NewDisk/ifpw_operate/trainset/rotate30',
              '/home/zhangyuqi/NewDisk/ifpw_operate/trainset/transpose']

train_path = '/home/zhangyuqi/NewDisk/train'
test_path = '/home/zhangyuqi/NewDisk/test'
valid_path = '/home/zhangyuqi/NewDisk/valid'

train_label = '/home/zhangyuqi/NewDisk/train_label1.txt'
test_label = '/home/zhangyuqi/NewDisk/test_label1.txt'
valid_label = '/home/zhangyuqi/NewDisk/valid_label1.txt'
# ---------------------------------------------------------------------------------------

# -----------------------------------for original images------------------------------------
def write_label(input_path, file_path):
	for filename in os.listdir(input_path):
		if filename.split('.')[-1] == 'png' or filename.split('.')[-1] == 'jpg':
			filename = input_path + '/' + filename
			make_one_txt(file_path, filename)

def make_one_txt(file_path, filename):
	num_of_line = 1
	input_path = filename.split('.')[0] + '.pts'
	image_path = filename
	print('outdoor_path:', input_path)
	with open(input_path, 'r') as f:
		while True:
			line = f.readline()
			line = line.strip('\n')
			print(line)
			if num_of_line == 1:
				write_txt(file_path, image_path)
			elif num_of_line > 3 and num_of_line < 72:
				write_txt(file_path, ' ')
				write_txt(file_path, line)
			elif num_of_line >= 72:
				write_txt(file_path, '\n')
				break
			num_of_line += 1

def write_txt(file_path, line):
	with open(file_path, 'a') as f:
		f.write(line)
	f.close()
# ----------------------------------------------------------------------------------------

# -------------------------------------for preprocessed images---------------------------

def write_label_prep():
	if not os.path.exists(train_path):
		os.mkdir(train_path)
	if not os.path.exists(test_path):
		os.mkdir(test_path)
	if not os.path.exists(valid_path):
		os.mkdir(valid_path)
	for path in input_path:
		image_path = path
		for filename in os.listdir(image_path):
			if filename.split('.')[-1] == 'jpg' or filename.split('.')[-1] == 'png':
				a_rand = random.uniform(0, 1)
				img_path = image_path + '/' +filename
				label_path = image_path + '/' + filename[:-4] + '.pts'
				if a_rand < perValid:
					img_aim_path = valid_path + '/' +filename
					label_aim_path = valid_label
				elif a_rand < perTest:
					img_aim_path = test_path + '/' +filename
					label_aim_path = test_label
				else:
					img_aim_path = train_path + '/' +filename
					label_aim_path = train_label
				img = cv2.imread(img_path)
				#height, width = img.shape[:2]
				#img_resize = cv2.resize(img, (img_w, img_h))
				#cv2.imwrite(img_aim_path, img_resize)
				num_of_line = 1
				landmark = []
				with open(label_path, 'r') as f:
					while True:
						line = f.readline()
						line = line.strip('\n')
						print(line)
						if num_of_line == 1:
							write_txt(label_aim_path, img_aim_path)
						elif num_of_line > 3 and num_of_line < 72:
							#write_txt(label_aim_path, ' ')
							#x = float(line.split()[0])
							landmark.append(line.split()[0])
							#y = float(line.split()[1])
							landmark.append(line.split()[1])
							#x = round(x * img_w / width)
							#y = round(y * img_h / height)
							#new_line = str(x) + ' ' + str(y)
							#write_txt(label_aim_path, new_line)
						elif num_of_line >= 72:
							#write_txt(label_aim_path, '\n')
							break
						num_of_line += 1
				cropimg, lk = crop_img(img, landmark)
				height, width = cropimg.shape[:2]
				cropimg_resize = cv2.resize(cropimg, (img_w, img_h))
				cv2.imwrite(img_aim_path, cropimg_resize)
				ldtxt = ''
				for i in range(68):
					x = round(lk[i*2] * img_w / width)
					lk[i*2] = round(lk[i*2] * img_w / width)
					y = round(lk[i*2+1] * img_h / height)
					lk[i * 2 + 1] = round(lk[i*2+1] * img_h / height)
					ldtxt = ldtxt + ' ' + str(x) + ' ' + str(y)
				ldtxt = ldtxt + '\n'
				write_txt(label_aim_path, ldtxt)





if __name__ == '__main__':
	# ---------------------------------for original images--------------------------------
	'''write_label(afw_path, afw_file)
	write_label(helen_test_path, helen_test_file)
	write_label(helen_train_path, helen_train_file)
	write_label(ibug_path, ibug_file)
	write_label(ifpw_test_path, ifpw_test_file)
	write_label(ifpw_train_path, ifpw_train_file)'''
	# ----------------------------for preprocessed images---------------------------------

	write_label_prep()



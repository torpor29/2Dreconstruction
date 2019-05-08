# divide celebA dataset
import sys
sys.path.append('/util')
import os
import random
import shutil
import cv2

def main():
	raw_txt = '/home/zhangyuqi/clone/facedetect/0_raw_data/Data/celeba_label.txt'
	relative_path = '/home/zhangyuqi/clone/facedetect/0_raw_data/Data/img_align_celeba/'  # for  find the img
	train_txt = '/home/zhangyuqi/projects/dataset/raw_train_label.txt'  # target txt
	valid_txt = '/home/zhangyuqi/projects/dataset/raw_valid_label.txt'
	test_txt = '/home/zhangyuqi/projects/dataset/raw_test_label.txt'
	train_img_fold = '/home/zhangyuqi/projects/dataset/train/'
	valid_img_fold = '/home/zhangyuqi/projects/dataset/valid/'
	test_img_fold = '/home/zhangyuqi/projects/dataset/test/'

	perTrain = 0.7  # percentage of train set
	perValid = 0.1
	perTest = 0.2
	line_num = 0
	train_num = 0
	test_num = 0
	valid_num = 0
	netw = 160
	neth = 160
	n_p = 5
	train_f = open(train_txt,"w")
	test_f = open(test_txt,"w")
	valid_f = open(valid_txt, "w")

	for line in open(raw_txt):
		if line.isspace() : continue  # skip empty line
		line_num += 1
		img_name = line.split()[0]
		full_img_path = relative_path + img_name
		a_rand = random.uniform(0,1)
		# train set
		if a_rand <= perValid:
			valid_img_path = valid_img_fold + img_name
			img = cv2.imread(full_img_path)
			w = img.shape[1]
			h = img.shape[0]
			raw_land = list(line.split())[1:2*n_p+1]
			new_line = img_name
			for i in range(n_p):
				x_ = round((float(raw_land[2*i+0]))*netw/w)
				y_ = round((float(raw_land[2*i+1]))*neth/h)
				new_line = new_line + ' ' + str(x_)
				new_line = new_line + ' ' + str(y_)
			# shutil.copy(full_img_path, train_img_path)
			new_line = valid_img_fold + new_line
			print('new_line', new_line)
			valid_f.write(new_line + '\n')

			valid_num += 1
			scale_img = cv2.resize(img, (netw, neth))
			if not os.path.exists(valid_img_fold):
				os.makedirs(valid_img_fold)
			print('valid output path', valid_img_path)
			cv2.imwrite(valid_img_path, scale_img)

		elif a_rand <= perTest:
			test_img_path = test_img_fold + img_name
			img = cv2.imread(full_img_path)
			w = img.shape[1]
			h = img.shape[0]
			raw_land = list(line.split())[1:2 * n_p + 1]
			new_line = img_name
			for i in range(n_p):
				x_ = round((float(raw_land[2 * i + 0])) * netw / w)
				y_ = round((float(raw_land[2 * i + 1])) * neth / h)
				new_line = new_line + ' ' + str(x_)
				new_line = new_line + ' ' + str(y_)
			# shutil.copy(full_img_path, train_img_path)
			new_line = test_img_fold + new_line
			print('new_line', new_line)
			test_f.write(new_line + '\n')

			test_num += 1
			scale_img = cv2.resize(img, (netw, neth))
			if not os.path.exists(test_img_fold):
				os.makedirs(test_img_fold)
			print('test output path', test_img_path)
			cv2.imwrite(test_img_path, scale_img)
		# test set
		else:
			train_img_path = train_img_fold + img_name
			img = cv2.imread(full_img_path)
			w = img.shape[1]
			h = img.shape[0]
			raw_land = list(line.split())[1:2 * n_p + 1]
			new_line = img_name
			for i in range(n_p):
				x_ = round((float(raw_land[2 * i + 0])) * netw / w)
				y_ = round((float(raw_land[2 * i + 1])) * neth / h)
				new_line = new_line + ' ' + str(x_)
				new_line = new_line + ' ' + str(y_)
			# shutil.copy(full_img_path, train_img_path)
			new_line = train_img_fold + new_line
			print('new_line', new_line)
			train_f.write(new_line + '\n')

			train_num += 1
			scale_img = cv2.resize(img, (netw, neth))
			if not os.path.exists(train_img_fold):
				os.makedirs(train_img_fold)
			print('train output path', train_img_path)
			cv2.imwrite(train_img_path, scale_img)
		print('img : ', line_num)

	train_f.close()
	test_f.close()
	valid_f.close()

if __name__ == '__main__':
	main()


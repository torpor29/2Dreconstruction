#rewrite the landmark to label.txt file(afw, helen, ibug, ifpw)

import math
import os

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

if __name__ == '__main__':
	write_label(afw_path, afw_file)
	write_label(helen_test_path, helen_test_file)
	write_label(helen_train_path, helen_train_file)
	write_label(ibug_path, ibug_file)
	write_label(ifpw_test_path, ifpw_test_file)
	write_label(ifpw_train_path, ifpw_train_file)


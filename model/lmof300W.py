#rewrite the landmark to label.txt file(300W)

import math
import os

Windoor_path = '/home/zhangyuqi/NewDisk/300W/01_Indoor/'
Woutdoor_path = '/home/zhangyuqi/NewDisk/300W/02_Outdoor/'
Windoor_file = '/home/zhangyuqi/projects/model/indoor_label_raw.txt'
Woutdoor_file = '/home/zhangyuqi/projects/model/outdoor_label_raw.txt'


def make_txt(totol_num):

	tmp_index = 1
	while tmp_index <= totol_num:
		if tmp_index < 10:
			index = '00' + str(tmp_index)
		elif tmp_index < 100:
			index = '0' + str(tmp_index)
		else:
			index = str(tmp_index)
		make_one_txt_outdoor(index)
		make_one_txt_indoor(index)
		tmp_index = tmp_index + 1



def make_one_txt_outdoor(index):
	num_of_line = 1
	input_path = Woutdoor_path + 'outdoor_{}.pts'.format(index)
	image_path = Woutdoor_path + 'outdoor_{}.png'.format(index)
	print('outdoor_path:', input_path)
	with open(input_path, 'r') as f:
		while True:
			line = f.readline()
			line = line.strip('\n')
			print(line)
			if num_of_line == 1:
				write_txt(Woutdoor_file, image_path)

			elif num_of_line > 3 and num_of_line < 72:
				write_txt(Woutdoor_file, ' ')
				write_txt(Woutdoor_file, line)
			elif num_of_line >= 72:
				write_txt(Woutdoor_file, '\n')
				break
			num_of_line += 1

def make_one_txt_indoor(index):
	num_of_line = 1
	input_path = Windoor_path + 'indoor_{}.pts'.format(index)
	image_path = Windoor_path + 'indoor_{}.png'.format(index)
	print('indoor_path:', input_path)
	with open(input_path, 'r') as f:
		while True:
			line = f.readline()
			line = line.strip('\n')
			print(line)
			if num_of_line == 1:
				write_txt(Windoor_file, image_path)
			elif num_of_line > 3 and num_of_line < 72:
				write_txt(Windoor_file, ' ')
				write_txt(Windoor_file, line)
			elif num_of_line >= 72:
				write_txt(Windoor_file, '\n')
				break
			num_of_line += 1

def write_txt(file_path, line):
	with open(file_path, 'a') as f:
		f.write(line)
	f.close()

if __name__ == '__main__':
	totol_num = 300
	make_txt(totol_num)



# augment image dataset
import os
import shutil
from diagnose_logging import Logger
from PIL import Image
import cv2
import numpy as np
import math

log = Logger('dataAug.py')
logger = log.getlog()

class ImgAug:
	def __init__(self, rootPath, export_path):
		self.rootPath = rootPath
		self.export_path = export_path

		try:
			if not os.path.exists(export_path):
				os.mkdir(export_path)
		except Exception as e:
			logger.error(e)
		logger.info('ImgAug: %s', rootPath)

	def get_savename(self, operate):
		"""

		:param operate: operations including transpose, rotation etc.
		:return: save path
		"""
		try:
			export_path = self.export_path
			out_path = export_path + '/' + operate

			if not os.path.exists(out_path):
				os.mkdir(out_path)

			savename = out_path + '/'

			logger.info('save:%s', savename)
			return savename

		except Exception as e:
			logger.error('get_savename ERROR')
			logger.error(e)

	def lightness(self, light):
		"""

		:param light: 0.87, 1.07
		:return:
		"""

		try:
			operate = 'lightness_' + str(light)
			# whole path
			rootPath = self.rootPath
			savepath = self.get_savename(operate)
			for filename in os.listdir(rootPath):
				if filename.split('.')[-1]=='png' or filename.split('.')[-1]=='jpg':
					with Image.open(rootPath + '/' + filename) as image:
						# change light of image
						out = image.point(lambda p: p * light)
						# rename
						outpath = savepath + filename.split('.')[0] + '_' + operate + '.' + filename.split('.')[-1]
						out.save(outpath)
					shutil.copy(rootPath + '/' + filename.split('.')[0] + '.' + 'pts',savepath)
		except Exception as e:
			logger.error('ERROR %s', operate)
			logger.error(e)

	def rotate(self, angle):
		"""

		:param angle: 15, 30
		:return:
		"""
		try:
			operate = 'rotate' + str(angle)

			rootPath = self.rootPath
			savepath = self.get_savename(operate)
			for filename in os.listdir(rootPath):
				if filename.split('.')[-1] == 'png' or filename.split('.')[-1] == 'jpg':
					img_raw = cv2.imread(rootPath + '/' + filename)
					height, width = img_raw.shape[:2]
					center = (width / 2, height / 2)
					scale = 1
					rangle = np.deg2rad(angle) # angle in radians
					# calculate new image width and height
					nw = (abs(np.sin(rangle) * height) + abs(np.cos(rangle) * width)) * scale
					nh = (abs(np.cos(rangle) * height) + abs(np.sin(rangle) * width)) * scale
					rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
					rot_move = np.dot(rot_mat, np.array([(nw - width) * 0.5, (nh - height) * 0.5, 0]))
					rot_mat[0, 2] += rot_move[0]
					rot_mat[1, 2] += rot_move[1]
					img_rotate = cv2.warpAffine(img_raw, rot_mat, (int(np.math.ceil(nw)), int(np.math.ceil(nh))),
					                            cv2.INTER_LANCZOS4, cv2.BORDER_REFLECT, 1)
					offset_w = (nw - width) / 2
					offset_h = (nh - height) / 2

					img_rotate = cv2.resize(img_rotate, (height,width))
					rw = width / nw
					rh = height / nh
					outpath = savepath + filename.split('.')[0] + '_' + operate + '.' + filename.split('.')[-1]
					#img_rotate.imwrite(outpath)# rename
					cv2.imwrite(outpath, img_rotate)
					with open(rootPath + '/' + filename.split('.')[0] + '.' + 'pts', 'r') as f:
						num_of_line = 1
						fa = open(savepath + filename.split('.')[0] + '_' + operate + '.pts', 'w')

						while True:
							line = f.readline()
							line = line.strip('\n')
							print(line)
							if num_of_line <= 3:
								fa.write(line)
								fa.write('\n')
							elif num_of_line > 3 and num_of_line < 72:
								x_raw = float(line.split()[0])
								y_raw = float(line.split()[1])
								(center_x, center_y) = center
								center_y = height - center_y
								x = (x_raw - center_x) * math.cos(rangle) - (y_raw - center_y) * math.sin(rangle) + center_x
								y = (x_raw - center_x) * math.sin(rangle) + (y_raw - center_y) * math.cos(rangle) + center_y
								x = round((x+offset_w) * rw, 6)
								y = round((height - y + offset_h) * rh, 6)
								fa.write(str(x) + ' ' + str(y) + '\n')
							elif num_of_line >= 72:
								fa.write(line)
								break
							num_of_line += 1
		except Exception as e:
			logger.error('ERROR %s', operate)
			logger.error(e)

	def transpose(self):
		# transpose the image
		try:
			operate = 'transpose'
			rootPath = self.rootPath
			savepath = self.get_savename(operate)
			for filename in os.listdir(rootPath):
				if filename.split('.')[-1] == 'png' or filename.split('.')[-1] == 'jpg':
					img_raw = cv2.imread(rootPath + '/' + filename)
					height, width = img_raw.shape[:2]

					img_transpose = cv2.flip(img_raw, 1)
					outpath = savepath + filename.split('.')[0] + '_' + operate + '.' + filename.split('.')[-1]
					# img_rotate.imwrite(outpath)# rename
					cv2.imwrite(outpath, img_transpose)
					with open(rootPath + '/' + filename.split('.')[0] + '.' + 'pts', 'r') as f:
						num_of_line = 1
						fa = open(savepath + filename.split('.')[0] + '_' + operate + '.pts', 'w')
						newpoints = []
						while True:
							line = f.readline()
							line = line.strip('\n')
							print(line)
							if num_of_line <= 3:
								fa.write(line)
								fa.write('\n')
							elif num_of_line > 3 and num_of_line < 72:
								x_raw = float(line.split()[0])
								y_raw = float(line.split()[1])
								x = round(width - x_raw, 6)
								y = y_raw
								newpoints.append([x, y])
							elif num_of_line >= 72:
								for i in range(16, -1, -1):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								for i in range(26, 16, -1):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								for i in range(27, 31):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								for i in range(35, 30, -1):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								for i in range(45, 41, -1):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								for i in range(47, 45, -1):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								for i in range(39, 35, -1):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								for i in range(41, 39, -1):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								for i in range(54, 47, -1):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								for i in range(59, 54, -1):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								for i in range(64, 59, -1):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								for i in range(67, 64, -1):
									fa.write(str(newpoints[i][0]) + ' ' + str(newpoints[i][1]) + '\n')
								fa.write(line)
								break
							num_of_line += 1
		except Exception as e:
			logger.error('ERROR %s', operate)
			logger.error(e)





def test():
	# source path and aim path
	rootPath_afw = '/home/zhangyuqi/NewDisk/afw'
	export_path_afw = '/home/zhangyuqi/NewDisk/afw_operate'
	imgPre_afw = ImgAug(rootPath_afw, export_path_afw)
	imgPre_afw.lightness(1.07)
	imgPre_afw.lightness(0.87)
	imgPre_afw.rotate(15)
	imgPre_afw.rotate(30)
	imgPre_afw.transpose()

	rootPath_ibug = '/home/zhangyuqi/NewDisk/ibug'
	export_path_ibug = '/home/zhangyuqi/NewDisk/ibug_operate'
	imgPre_ibug = ImgAug(rootPath_ibug, export_path_ibug)
	imgPre_ibug.lightness(1.07)
	imgPre_ibug.lightness(0.87)
	imgPre_ibug.rotate(15)
	imgPre_ibug.rotate(30)
	imgPre_ibug.transpose()

	rootPath_300W_in = '/home/zhangyuqi/NewDisk/300W/01_Indoor'
	export_path_300W_in = '/home/zhangyuqi/NewDisk/300W_operate/01_Indoor'
	rootPath_300W_out = '/home/zhangyuqi/NewDisk/300W/02_Outdoor'
	export_path_300W_out = '/home/zhangyuqi/NewDisk/300W_operate/02_Outdoor'
	imgPre_300W_in = ImgAug(rootPath_300W_in, export_path_300W_in)
	imgPre_300W_in.lightness(1.07)
	imgPre_300W_in.lightness(0.87)
	imgPre_300W_in.rotate(15)
	imgPre_300W_in.rotate(30)
	imgPre_300W_in.transpose()

	imgPre_300W_out = ImgAug(rootPath_300W_out, export_path_300W_out)
	imgPre_300W_out.lightness(1.07)
	imgPre_300W_out.lightness(0.87)
	imgPre_300W_out.rotate(30)
	imgPre_300W_out.rotate(15)
	imgPre_300W_out.transpose()

	rootPath_helen_test = '/home/zhangyuqi/NewDisk/helen/testset'
	export_path_helen_test = '/home/zhangyuqi/NewDisk/helen_operate/testset'
	rootPath_helen_train = '/home/zhangyuqi/NewDisk/helen/trainset'
	export_path_helen_train = '/home/zhangyuqi/NewDisk/helen_operate/trainset'
	imgPre_helen_test = ImgAug(rootPath_helen_test, export_path_helen_test)
	imgPre_helen_test.lightness(1.07)
	imgPre_helen_test.lightness(0.87)
	imgPre_helen_test.rotate(30)
	imgPre_helen_test.rotate(15)
	imgPre_helen_test.transpose()

	imgPre_helen_train = ImgAug(rootPath_helen_train, export_path_helen_train)
	imgPre_helen_train.lightness(1.07)
	imgPre_helen_train.lightness(0.87)
	imgPre_helen_train.rotate(30)
	imgPre_helen_train.rotate(15)
	imgPre_helen_train.transpose()

	rootPath_ifpw_test = '/home/zhangyuqi/NewDisk/ifpw/testset'
	export_path_ifpw_test = '/home/zhangyuqi/NewDisk/ifpw_operate/testset'
	rootPath_ifpw_train = '/home/zhangyuqi/NewDisk/ifpw/trainset'
	export_path_ifpw_train = '/home/zhangyuqi/NewDisk/ifpw_operate/trainset'
	imgPre_ifpw_test = ImgAug(rootPath_ifpw_test, export_path_ifpw_test)
	imgPre_ifpw_test.lightness(1.07)
	imgPre_ifpw_test.lightness(0.87)
	imgPre_ifpw_test.rotate(30)
	imgPre_ifpw_test.rotate(15)
	imgPre_ifpw_test.transpose()

	imgPre_ifpw_train = ImgAug(rootPath_ifpw_train, export_path_ifpw_train)
	imgPre_ifpw_train.lightness(1.07)
	imgPre_ifpw_train.lightness(0.87)
	imgPre_ifpw_train.rotate(30)
	imgPre_ifpw_train.rotate(15)
	imgPre_ifpw_train.transpose()


if __name__ == '__main__':
	import  datetime
	print('start...')
	start_time = datetime.datetime.now()

	test()

	end_time = datetime.datetime.now()
	time_consume = (end_time - start_time).microseconds / 1000000

	logger.info('start_time: %s', start_time)
	logger.info('end_time: %s', end_time)
	logger.info('time_consume: %s', time_consume)

	logger.info('main finish')






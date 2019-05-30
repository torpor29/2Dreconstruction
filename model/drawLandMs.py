# draw landmarks to see the result of image augment

import os
import sys
import cv2
import numpy as np


def drawpoint(rootPath, exportPath):
	if not os.path.exists(exportPath):
		os.mkdir(exportPath)
	for filename in os.listdir(rootPath):
		if filename.split('.')[-1] == 'jpg' or filename.split('.') == 'png':
			imgpath = rootPath + '/' + filename
			labelpath = rootPath + '/' + filename[:-4] + '.pts'
			img = cv2.imread(imgpath)
			draw_img = img.copy()

			lines = []
			for line in open(labelpath):
				lines.append(line)

			points = lines[3:71]
			for i in range(0,68):
				x = int(float(points[i].split()[0]))
				y = int(float(points[i].split()[1].split('/n')[0]))
				cv2.circle(draw_img, (x, y), 2, (0, 0, 255))
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(draw_img, str(i), (x+3,y), font, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
			# output image
			outpath = exportPath + '/' + filename
			cv2.imwrite(outpath, draw_img)

if __name__=='__main__':
	afw_input = '/home/zhangyuqi/NewDisk/afw'
	afw_draw = '/home/zhangyuqi/NewDisk/afw_draw'
	drawpoint(afw_input, afw_draw)

	W300_input = '/home/zhangyuqi/NewDisk/300W'
	W300_draw = '/home/zhangyuqi/NewDisk/300W_draw'
	drawpoint(W300_input, W300_draw)

	afw_operate_lightness_0_input = '/home/zhangyuqi/NewDisk/afw_operate/lightness_0.87'
	afw_operate_lightness_0_draw = '/home/zhangyuqi/NewDisk/afw_operate/lightness_0.87_draw'
	drawpoint(afw_operate_lightness_0_input, afw_operate_lightness_0_draw)

	afw_operate_lightness_1_input = '/home/zhangyuqi/NewDisk/afw_operate/lightness_1.07'
	afw_operate_lightness_1_draw = '/home/zhangyuqi/NewDisk/afw_operate/lightness_1.07_draw'
	drawpoint(afw_operate_lightness_1_input, afw_operate_lightness_1_draw)

	afw_operate_rotate_15_input = '/home/zhangyuqi/NewDisk/afw_operate/rotate15'
	afw_operate_rotate_15_draw = '/home/zhangyuqi/NewDisk/afw_operate/rotate15_draw'
	drawpoint(afw_operate_rotate_15_input, afw_operate_rotate_15_draw)

	afw_operate_rotate_30_input = '/home/zhangyuqi/NewDisk/afw_operate/rotate30'
	afw_operate_rotate_30_draw = '/home/zhangyuqi/NewDisk/afw_operate/rotate30_draw'
	drawpoint(afw_operate_rotate_30_input, afw_operate_rotate_30_draw)

	afw_operate_transpose_input = '/home/zhangyuqi/NewDisk/afw_operate/transpose'
	afw_operate_transpose_draw = '/home/zhangyuqi/NewDisk/afw_operate/transpose_draw'
	drawpoint(afw_operate_transpose_input, afw_operate_transpose_draw)



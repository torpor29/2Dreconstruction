import os
import matplotlib
from matplotlib import pyplot as plt
import cv2
import numpy as np

def show_landmarks(image, landMarks):
	"""
	show image landmarks
	:param image:
	:param landMarks:
	:return:
	"""
	showPoints = []
	for i in range(round(len(landMarks)/2)):
		point = []
		point.append(landMarks[i*2+0])
		point.append(landMarks[i*2+1])
		showPoints.append(point)

	showPoints = np.array(showPoints)
	image = image[:, :, [2, 1, 0]]
	plt.imshow(image)
	plt.scatter(showPoints[:, 0], showPoints[:, 1], marker=".", c="r")
	plt.show()


def main():
	imagepath_txt = '/home/zhangyuqi/projects/model/tmp/2019-05-06_22-49-32/test_label.txt'
	imagepath_f = open(imagepath_txt, 'r')
	count = 0
	for line in imagepath_f:

		if count >= 20:
			break
		image_path = line.split()[0]
		img = cv2.imread(image_path)
		landmark=[]
		for i in range(1, len(line.split())):
			landmark.append(float(line.split()[i]) * 80 + 80)
		show_landmarks(img, landmark)
		count += 1

	imagepath_f.close()

if __name__ == '__main__':
    main()
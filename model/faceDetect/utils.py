import numpy as np

def IoU(box, boxes):
	'''
	to calculate IoU between detect box and gt boxes
	:param box: input box, np array, shape(5,:):x1, y1, x2, y2, score
	:param boxes: gt boxes, np array, shape(n, 4): x1, y1, x2, y2
	:return: the IoU np array, shape(n, :)
	'''

	box_area = (box[2] - box[0] +1) * (box[3] - box[1] + 1)
	area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
	xx1 = np.maximum(box[0], boxes[:, 0])
	yy1 = np.maximum(box[1], boxes[:, 1])
	xx2 = np.minimum(box[2], boxes[:, 2])
	yy2 = np.minimum(box[3], boxes[:, 3])

	w = np.maximum(0, xx2 - xx1 + 1)
	h = np.maximum(0, yy2 - yy1 + 1)

	inter = w * h
	ovr = np.true_divide(inter, (box_area + area -inter))
	return ovr


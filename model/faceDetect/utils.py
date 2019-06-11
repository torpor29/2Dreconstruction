import numpy as np
import torchvision.transforms as transforms
import torch
from torch.autograd.variable import Variable
from matplotlib.patches import Circle

transform = transforms.ToTensor()

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


def convert_image_to_tensor(image):
	'''

	:param image: numpy array, h*w*c
	:return: pytorch.FloatTensor, c*h*w
	'''
	image = image.astype(np.float)
	return transform(image)

def convert_chwtensor_to_hwcnumpy(tensor):
	if isinstance(tensor, Variable):
		return np.transpose(tensor.data.numpy(), (0, 2, 3, 1))
	elif isinstance(tensor, torch.FloatTensor):
		return np.transpose(tensor.numpy(), (0, 2, 3, 1))
	else:
		raise Exception("covert b*c*h*w tensor to b*h*w*c numpy error.This tensor must have 4 dimension.")

def nms(dets, thresh, mode='Union'):
	"""
	    greedily select boxes with high confidence
	    keep boxes overlap <= thresh
	    rule out overlap > thresh
	    :param dets: [[x1, y1, x2, y2 score]]
	    :param thresh: retain overlap <= thresh
	    :return: indexes to keep
	    """
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	scores = dets[:, 4]

	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]

	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])

		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		if mode == "Union":
			ovr = inter / (areas[i] + areas[order[1:]] - inter)
		elif mode == "Minimum":
			ovr = inter / np.minimum(areas[i], areas[order[1:]])

		inds = np.where(ovr <= thresh)[0]
		order = order[inds + 1]

	return keep

def vis_face(im_array, dets, landmarks=None):
	"""Visualize detection results before and after calibration

	    Parameters:
	    ----------
	    im_array: numpy.ndarray, shape(1, c, h, w)
	        test image in rgb
	    dets1: numpy.ndarray([[x1 y1 x2 y2 score]])
	        detection results before calibration
	    dets2: numpy.ndarray([[x1 y1 x2 y2 score]])
	        detection results after calibration
	    thresh: float
	        boxes with scores > thresh will be drawn in red otherwise yellow

	    Returns:
	    -------
	    """
	import matplotlib.pyplot as plt
	import random
	import pylab

	figure = pylab.figure()
	# plt.subplot(121)
	pylab.imshow(im_array)
	figure.suptitle('DFace Detector', fontsize=20)

	for i in range(dets.shape[0]):
		bbox = dets[i, :4]

		rect = pylab.Rectangle((bbox[0], bbox[1]),
		                       bbox[2] - bbox[0],
		                       bbox[3] - bbox[1], fill=False,
		                       edgecolor='yellow', linewidth=0.9)
		pylab.gca().add_patch(rect)
	pylab.show()

	if landmarks is not None:
		for i in range(landmarks.shape[0]):
			landmarks_one = landmarks[i, :]
			landmarks_one = landmarks_one.reshape((5, 2))
			for j in range(5):
				# pylab.scatter(landmarks_one[j, 0], landmarks_one[j, 1], c='yellow', linewidths=0.1, marker='x', s=5)

				cir1 = Circle(xy=(landmarks_one[j, 0], landmarks_one[j, 1]), radius=2, alpha=0.4, color="red")
				pylab.gca().add_patch(cir1)
			# plt.gca().text(bbox[0], bbox[1] - 2,
			#                '{:.3f}'.format(score),
			#                bbox=dict(facecolor='blue', alpha=0.5), fontsize=12, color='white')
			# else:
			#     rect = plt.Rectangle((bbox[0], bbox[1]),
			#                          bbox[2] - bbox[0],
			#                          bbox[3] - bbox[1], fill=False,
			#                          edgecolor=color, linewidth=0.5)
			#     plt.gca().add_patch(rect)

		pylab.show()

def convert_to_square(boxes_align):
	'''
	convert bbox to square
	:param boxes_align:
	:return:
	'''

	square_bbox = boxes_align.copy()
	#print(boxes_align.shape)
	#print('\n')
	h = boxes_align[:, 3] - boxes_align[:, 1] + 1
	w = boxes_align[:, 2] - boxes_align[:, 0] + 1
	max_side = np.maximum(h, w)
	square_bbox[:, 0] = boxes_align[:, 0] + w * 0.5 - max_side * 0.5
	square_bbox[:, 1] = boxes_align[:, 1] + h * 0.5 - max_side * 0.5
	square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
	square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
	return square_bbox
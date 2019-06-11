import cv2
import time
import numpy as np
import torch
from torch.autograd.variable import Variable
from faceDetectPnet import PNet
import utils
from utils import convert_chwtensor_to_hwcnumpy
from utils import convert_image_to_tensor

def creat_pnet(p_model_path = None, device = None):
	pnet = None
	if p_model_path is not None:
		pnet = PNet(is_train=True, use_cuda=True)
		pnet = torch.nn.DataParallel(pnet, [1])
		pnet.to(device=device, dtype=torch.float32)
		print("=> loading checkpoint '{}'".format(p_model_path))
		checkpoint = torch.load(p_model_path, map_location=device)
		pnet.load_state_dict(checkpoint['state_dict'])
	pnet.eval()

	return pnet

class PnetDetector(object):
	def __init__(self, pnet=None, min_face_size=12, stride=2, threshold=[0.6, 0.7, 0.7], scale_factor=0.709):
		self.pnet_detector = pnet
		self.min_face_size = min_face_size
		self.stride = stride
		self.thresh = threshold
		self.scale_factor = scale_factor

	def unique_image_format(self, im):
		if not isinstance(im, np.ndarray):
			if im.mode == 'I':
				im = np.array(im, np.int32, copy=False)
			elif im.mode == 'I;16':
				im = np.array(im, np.int16, copy=False)
			else:
				im = np.asarray(im)
		return im

	def square_bbox(self, bbox):
		"""
			convert bbox to square
		Parameters:
		----------
			bbox: numpy array , shape n x m
				input bbox
		Returns:
		-------
			square bbox
		"""
		square_bbox = bbox.copy()

		h = bbox[:, 3] - bbox[:, 1] + 1
		w = bbox[:, 2] - bbox[:, 0] + 1
		l = np.maximum(h, w)
		square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - l * 0.5
		square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - l * 0.5

		square_bbox[:, 2] = square_bbox[:, 0] + l - 1
		square_bbox[:, 3] = square_bbox[:, 1] + l - 1
		return square_bbox

	def generate_bounding_box(self, map, reg, scale, threshold):
		"""
			generate bbox from feature map
		Parameters:
		----------
			map: numpy array , n x m x 1
				detect score for each position
			reg: numpy array , n x m x 4
				bbox
			scale: float number
				scale of this detection
			threshold: float number
				detect threshold
		Returns:
		-------
			bbox array
		"""
		stride = 2
		cellsize = 12

		t_index = np.where(map > threshold)

		# find nothing
		if t_index[0].size == 0:
			return np.array([])

		dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i] for i in range(4)]
		reg = np.array([dx1, dy1, dx2, dy2])

		# lefteye_dx, lefteye_dy, righteye_dx, righteye_dy, nose_dx, nose_dy, \
		# leftmouth_dx, leftmouth_dy, rightmouth_dx, rightmouth_dy = [landmarks[0, t_index[0], t_index[1], i] for i in range(10)]
		#
		# landmarks = np.array([lefteye_dx, lefteye_dy, righteye_dx, righteye_dy, nose_dx, nose_dy, leftmouth_dx, leftmouth_dy, rightmouth_dx, rightmouth_dy])

		score = map[t_index[0], t_index[1], 0]
		boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
		                         np.round((stride * t_index[0]) / scale),
		                         np.round((stride * t_index[1] + cellsize) / scale),
		                         np.round((stride * t_index[0] + cellsize) / scale),
		                         score,
		                         reg,
		                         # landmarks
		                         ])

		return boundingbox.T

	def resize_image(self, img, scale):
		"""
			resize image and transform dimention to [batchsize, channel, height, width]
		Parameters:
		----------
			img: numpy array , height x width x channel
				input image, channels in BGR order here
			scale: float number
				scale factor of resize operation
		Returns:
		-------
			transformed image tensor , 1 x channel x height x width
		"""
		height, width, channels = img.shape
		new_height = int(height * scale)  # resized new height
		new_width = int(width * scale)  # resized new width
		new_dim = (new_width, new_height)
		img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
		return img_resized

	def detect_pnet(self, im):
		"""Get face candidates through pnet

		Parameters:
		----------
		im: numpy array
			input image array

		Returns:
		-------
		boxes: numpy array
			detected boxes before calibration
		boxes_align: numpy array
			boxes after calibration
		"""

		# im = self.unique_image_format(im)

		h, w, c = im.shape
		net_size = 12

		current_scale = float(net_size) / self.min_face_size  # find initial scale
		im_resized = self.resize_image(im, current_scale)
		current_height, current_width, _ = im_resized.shape

		# fcn
		all_boxes = list()
		while min(current_height, current_width) > net_size:
			feed_imgs = []
			image_tensor = convert_image_to_tensor(im_resized)
			feed_imgs.append(image_tensor)
			feed_imgs = torch.stack(feed_imgs)

			feed_imgs = feed_imgs.to(device='cuda: 1', dtype=torch.float32)

			cls_map, reg = self.pnet_detector(feed_imgs)

			cls_map_np = convert_chwtensor_to_hwcnumpy(cls_map.cpu())
			reg_np = convert_chwtensor_to_hwcnumpy(reg.cpu())
			# landmark_np = image_tools.convert_chwTensor_to_hwcNumpy(landmark.cpu())

			boxes = self.generate_bounding_box(cls_map_np[0, :, :], reg_np, current_scale, self.thresh[0])

			current_scale *= self.scale_factor
			im_resized = self.resize_image(im, current_scale)
			current_height, current_width, _ = im_resized.shape

			if boxes.size == 0:
				continue
			keep = utils.nms(boxes[:, :5], 0.5, 'Union')
			boxes = boxes[keep]
			all_boxes.append(boxes)

		if len(all_boxes) == 0:
			return None, None

		all_boxes = np.vstack(all_boxes)

		# merge the detection from first stage
		keep = utils.nms(all_boxes[:, 0:5], 0.7, 'Union')
		all_boxes = all_boxes[keep]
		# boxes = all_boxes[:, :5]

		bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
		bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

		# landmark_keep = all_boxes[:, 9:].reshape((5,2))

		boxes = np.vstack([all_boxes[:, 0],
		                   all_boxes[:, 1],
		                   all_boxes[:, 2],
		                   all_boxes[:, 3],
		                   all_boxes[:, 4],
		                   ])

		boxes = boxes.T

		align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
		align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
		align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
		align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

		# refine the boxes
		boxes_align = np.vstack([align_topx,
		                         align_topy,
		                         align_bottomx,
		                         align_bottomy,
		                         all_boxes[:, 4],
		                         ])
		boxes_align = boxes_align.T

		return boxes_align
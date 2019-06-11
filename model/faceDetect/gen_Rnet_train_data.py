# generate Rnet train data

import numpy as np
import cv2
import os
from utils import IoU
import config
import random
from detectFacePnet import creat_pnet, PnetDetector
from utils import convert_to_square


def gen_rnet_data(data_dir, anno_file, prefix, pnet_model_file):
	'''

	:param data_dir: train dataset dir
	:param anno_file: annotation file
	:param prefix: orign root category of rnet data
	:param pnet_model_file: pnet model
	:return:
	'''
	neg_save_dir = os.path.join(data_dir, '24_train/negative')
	pos_save_dir = os.path.join(data_dir, '24_train/positive')
	part_save_dir = os.path.join(data_dir, '24_train/part')

	neg_save_dir_val = os.path.join(data_dir, '24_val/negative')
	pos_save_dir_val = os.path.join(data_dir, '24_val/positive')
	part_save_dir_val = os.path.join(data_dir, '24_val/part')

	neg_save_dir_test = os.path.join(data_dir, '24_test/negative')
	pos_save_dir_test = os.path.join(data_dir, '24_test/positive')
	part_save_dir_test = os.path.join(data_dir, '24_test/part')

	per_train = 0.7
	per_val = 0.2
	per_test = 0.1

	image_size = 24



	for dir_path in [neg_save_dir, pos_save_dir, part_save_dir,
	                 neg_save_dir_val,pos_save_dir_val, part_save_dir_val,
	                 neg_save_dir_test, pos_save_dir_test, part_save_dir_test]:
		if not os.path.exists(dir_path):
			os.mkdir(dir_path)


	post_save_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_POSITIVE_ANNO_FILENAME)
	neg_save_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_NEGATIVE_ANNO_FILENAME)
	part_save_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_PART_ANNO_FILENAME)

	post_save_test_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_POSITIVE_TEST_ANNO_FILENAME)
	neg_save_test_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_NEGATIVE_TEST_ANNO_FILENAME)
	part_save_test_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_PART_TEST_ANNO_FILENAME)

	post_save_val_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_POSITIVE_VALID_ANNO_FILENAME)
	neg_save_val_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_NEGATIVE_VALID_ANNO_FILENAME)
	part_save_val_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_PART_VALID_ANNO_FILENAME)

	f1 = open(post_save_file, 'w')
	f2 = open(neg_save_file, 'w')
	f3 = open(part_save_file, 'w')

	f1_test = open(post_save_test_file, 'w')
	f2_test = open(neg_save_test_file, 'w')
	f3_test = open(part_save_test_file, 'w')

	f1_val = open(post_save_val_file, 'w')
	f2_val = open(neg_save_val_file, 'w')
	f3_val = open(part_save_val_file, 'w')

	with open(anno_file, 'r') as f:
		annotations = f.readlines()
		random.shuffle(annotations)

	num = len(annotations)

	pnet = creat_pnet(pnet_model_file, 'cuda: 1')
	pnetDetector = PnetDetector(pnet=pnet, min_face_size=12)

	p_idx = 0
	n_idx = 0
	d_idx = 0
	image_done = 0


	all_boxes = list()
	for annotation in annotations[:1000]:
		annotation = annotation.strip().split(' ')
		path = os.path.join(prefix, annotation[0])
		bbox = list(map(float, annotation[1:]))  # generate boxes randomly to get negtive images
		boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
		per = random.randint(0, 500)
		img = cv2.imread(path)
		boxes_align = pnetDetector.detect_pnet(img)
		if isinstance(boxes_align, tuple):
			continue
		all_boxes.append(boxes_align)
		if image_done % 100 == 0:
			print("%d images done" % image_done)
		image_done += 1
		dets = convert_to_square(boxes_align)
		dets[:, 0:4] = np.round(dets[:, 0:4])
		for box in dets:
			x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
			width = x_right - x_left + 1
			height = y_bottom - y_top + 1

			if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
				continue

			Iou = IoU(box, boxes)
			cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
			resized_im = cv2.resize(cropped_im, (image_size, image_size),
			                        interpolation=cv2.INTER_LINEAR)

			# save negative images and write label
			if np.max(Iou) < 0.3:
				# Iou with all gts must below 0.3
				if per < 100:
					save_file = os.path.join(neg_save_dir_test, "%s.jpg" % n_idx)
					f2_test.write(save_file + ' 0\n')
					cv2.imwrite(save_file, resized_im)
				elif per < 300:
					save_file = os.path.join(neg_save_dir_val, "%s.jpg" % n_idx)
					f2_val.write(save_file + ' 0\n')
					cv2.imwrite(save_file, resized_im)
				else:
					save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
					f2.write(save_file + ' 0\n')
					cv2.imwrite(save_file, resized_im)
				n_idx += 1
			else:
				# find gt_box with the highest iou
				idx = np.argmax(Iou)
				assigned_gt = boxes[idx]
				x1, y1, x2, y2 = assigned_gt

				# compute bbox reg label
				offset_x1 = (x1 - x_left) / float(width)
				offset_y1 = (y1 - y_top) / float(height)
				offset_x2 = (x2 - x_right) / float(width)
				offset_y2 = (y2 - y_bottom) / float(height)

				# save positive and part-face images and write labels
				if np.max(Iou) >= 0.65:
					if per < 100:
						save_file = os.path.join(pos_save_dir_test, "%s.jpg" % p_idx)
						f1_test.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
							offset_x1, offset_y1, offset_x2, offset_y2))
						cv2.imwrite(save_file, resized_im)
					elif per < 300:
						save_file = os.path.join(pos_save_dir_val, "%s.jpg" % p_idx)
						f1_val.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
							offset_x1, offset_y1, offset_x2, offset_y2))
						cv2.imwrite(save_file, resized_im)
					else:
						save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
						f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
							offset_x1, offset_y1, offset_x2, offset_y2))
						cv2.imwrite(save_file, resized_im)
					p_idx += 1

				elif np.max(Iou) >= 0.4:
					if per < 100:
						save_file = os.path.join(part_save_dir_test, "%s.jpg" % p_idx)
						f3_test.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
							offset_x1, offset_y1, offset_x2, offset_y2))
						cv2.imwrite(save_file, resized_im)
					elif per < 300:
						save_file = os.path.join(part_save_dir_val, "%s.jpg" % p_idx)
						f3_val.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
							offset_x1, offset_y1, offset_x2, offset_y2))
						cv2.imwrite(save_file, resized_im)
					else:
						save_file = os.path.join(part_save_dir, "%s.jpg" % p_idx)
						f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
							offset_x1, offset_y1, offset_x2, offset_y2))
						cv2.imwrite(save_file, resized_im)
					d_idx += 1
	f1.close()
	f2.close()
	f3.close()
	f1_val.close()
	f2_val.close()
	f3_val.close()
	f1_test.close()
	f2_test.close()
	f3_test.close()



if __name__ == '__main__':
	data_dir = '/home/zhangyuqi/NewDisk/widerFace_train'
	if not os.path.exists(data_dir):
		os.mkdir(data_dir)
	anno_file = os.path.join(config.ANNO_STORE_DIR, 'wider_origin_anno.txt')
	prefix_path = '/home/zhangyuqi/NewDisk/WIDER_train/images'
	pnet_model_file = '/home/zhangyuqi/projects/model/faceDetect/results/2019-06-03_12-01-37/checkpoint.pth.tar'

	gen_rnet_data(data_dir, anno_file, prefix_path, pnet_model_file)


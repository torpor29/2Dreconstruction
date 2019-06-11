import os
import numpy.random as npr
import numpy as np

def assemble_data(output_file, anno_file_list = []):
	'''
	assemble the annotations to one file to be trained
	size = 12
	:param output_file: new annotation file
	:param anno_file_list:  neg_anno_file, pos_anno_file, part_anno_file
	:return:
	'''

	size = 12

	if len(anno_file_list) == 0:
		return 0

	if os.path.exists(output_file):
		os.remove(output_file)

	for anno_file in anno_file_list:
		with open(anno_file, 'r') as f:
			anno_lines = f.readlines()

		base_num = 5000 #set the threshold of choosing annotations randomly

		if len(anno_lines) > base_num * 3:
			idx_keep = npr.choice(len(anno_lines), size=base_num * 3, replace=True)
		elif len(anno_lines) < 5000:
			idx_keep = npr.choice(len(anno_lines), size= len(anno_lines), replace=True)
		else:
			idx_keep = np.arange(len(anno_lines))
			np.random.shuffle(idx_keep)
		chose_count = 0
		with open(output_file, 'a+') as f:
			for idx in idx_keep:
				f.write(anno_lines[idx])
				chose_count += 1
	return chose_count

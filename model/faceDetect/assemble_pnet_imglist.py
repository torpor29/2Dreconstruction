import os
import config
import assemble

if __name__ == '__main__':
	anno_list = []

	pnet_pos_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_POSITIVE_ANNO_FILENAME)
	pnet_part_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_PART_ANNO_FILENAME)
	pnet_neg_file = os.path.join(config.ANNO_STORE_DIR, config.PNET_NEGATIVE_ANNO_FILENAME)

	anno_list.append(pnet_pos_file)
	anno_list.append(pnet_part_file)
	anno_list.append(pnet_neg_file)

	imglist_filename = config.PNET_TRAIN_IMGLIST_FILENAME
	anno_dir = config.ANNO_STORE_DIR
	imglist_file = os.path.join(anno_dir, imglist_filename)

	chose_count = assemble.assemble_data(imglist_file, anno_list)

	print('PNet train annotation result file path: .'.format(imglist_file))
import os
import config
import assemble

if __name__ == '__main__':
	anno_list = []
	anno_list_test = []
	anno_list_val = []

	rnet_pos_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_POSITIVE_ANNO_FILENAME)
	rnet_part_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_PART_ANNO_FILENAME)
	rnet_neg_file = os.path.join(config.ANNO_STORE_DIR, config.RNET_NEGATIVE_ANNO_FILENAME)

	rnet_pos_file_test = os.path.join(config.ANNO_STORE_DIR, config.RNET_POSITIVE_TEST_ANNO_FILENAME)
	rnet_part_file_test = os.path.join(config.ANNO_STORE_DIR, config.RNET_PART_TEST_ANNO_FILENAME)
	rnet_neg_file_test = os.path.join(config.ANNO_STORE_DIR, config.RNET_NEGATIVE_TEST_ANNO_FILENAME)

	rnet_pos_file_val = os.path.join(config.ANNO_STORE_DIR, config.RNET_POSITIVE_VALID_ANNO_FILENAME)
	rnet_part_file_val = os.path.join(config.ANNO_STORE_DIR, config.RNET_PART_VALID_ANNO_FILENAME)
	rnet_neg_file_val = os.path.join(config.ANNO_STORE_DIR, config.RNET_NEGATIVE_VALID_ANNO_FILENAME)

	anno_list.append(rnet_pos_file)
	anno_list.append(rnet_part_file)
	anno_list.append(rnet_neg_file)

	anno_list_test.append(rnet_pos_file_test)
	anno_list_test.append(rnet_part_file_test)
	anno_list_test.append(rnet_neg_file_test)

	anno_list_val.append(rnet_pos_file_val)
	anno_list_val.append(rnet_part_file_val)
	anno_list_val.append(rnet_neg_file_val)

	imglist_filename = config.RNET_TRAIN_IMGLIST_FILENAME
	anno_dir = config.ANNO_STORE_DIR
	imglist_file = os.path.join(anno_dir, imglist_filename)

	imglist_filename_test = config.RNET_TRAIN_IMGLIST_FILENAME_TEST
	anno_dir_test = config.ANNO_STORE_DIR
	imglist_file_test = os.path.join(anno_dir, imglist_filename_test)

	imglist_filename_val = config.RNET_TRAIN_IMGLIST_FILENAME_VALID
	anno_dir_val = config.ANNO_STORE_DIR
	imglist_file_val = os.path.join(anno_dir, imglist_filename_val)

	chose_count = assemble.assemble_data(imglist_file, anno_list)
	chose_count_test = assemble.assemble_data(imglist_file_test, anno_list_test)
	chose_count_val = assemble.assemble_data(imglist_file_val, anno_list_val)

	print('PNet train annotation result file path: {}'.format(imglist_file))
	print('PNet test annotation result file path: {}'.format(imglist_file_test))
	print('PNet valid annotation result file path: {}'.format(imglist_file_val))
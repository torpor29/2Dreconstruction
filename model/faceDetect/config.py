import os

root_path = '/home/zhangyuqi/projects/model/faceDetect'

MODEL_STORE_DIR = os.path.join(root_path, 'model_store')

ANNO_STORE_DIR = os.path.join(root_path, 'anno_store')

LOG_DIR = os.path.join(root_path, 'log')

USE_CUDA = True

TRAIN_BATCH_SIZE = 256

TRAIN_LR = 0.001

END_EPOCH = 100

ORIGIN_ANNO_FILE = '/home/zhangyuqi/projects/model/faceDetect/anno_store/wider_origin_anno.txt'

PNET_POSITIVE_ANNO_FILENAME = 'pos_12.txt'
PNET_NEGATIVE_ANNO_FILENAME = "neg_12.txt"
PNET_PART_ANNO_FILENAME = "part_12.txt"
PNET_LANDMARK_ANNO_FILENAME = "landmark_12.txt"

PNET_POSITIVE_TEST_ANNO_FILENAME = 'pos_test_12.txt'
PNET_NEGATIVE_TEST_ANNO_FILENAME = "neg_test_12.txt"
PNET_PART_TEST_ANNO_FILENAME = "part_test_12.txt"
PNET_LANDMARK_TEST_ANNO_FILENAME = "landmark_test_12.txt"

PNET_POSITIVE_VALID_ANNO_FILENAME = 'pos_valid_12.txt'
PNET_NEGATIVE_VALID_ANNO_FILENAME = "neg_valid_12.txt"
PNET_PART_VALID_ANNO_FILENAME = "part_valid_12.txt"
PNET_LANDMARK_VALID_ANNO_FILENAME = "landmark_valid_12.txt"


RNET_POSITIVE_ANNO_FILENAME = "pos_24.txt"
RNET_NEGATIVE_ANNO_FILENAME = "neg_24.txt"
RNET_PART_ANNO_FILENAME = "part_24.txt"
RNET_LANDMARK_ANNO_FILENAME = "landmark_24.txt"

RNET_POSITIVE_TEST_ANNO_FILENAME = 'pos_test_24.txt'
RNET_NEGATIVE_TEST_ANNO_FILENAME = "neg_test_24.txt"
RNET_PART_TEST_ANNO_FILENAME = "part_test_24.txt"
RNET_LANDMARK_TEST_ANNO_FILENAME = "landmark_test_24.txt"

RNET_POSITIVE_VALID_ANNO_FILENAME = 'pos_valid_24.txt'
RNET_NEGATIVE_VALID_ANNO_FILENAME = "neg_valid_24.txt"
RNET_PART_VALID_ANNO_FILENAME = "part_valid_24.txt"
RNET_LANDMARK_VALID_ANNO_FILENAME = "landmark_valid_24.txt"


ONET_POSITIVE_ANNO_FILENAME = "pos_48.txt"
ONET_NEGATIVE_ANNO_FILENAME = "neg_48.txt"
ONET_PART_ANNO_FILENAME = "part_48.txt"
ONET_LANDMARK_ANNO_FILENAME = "landmark_48.txt"

ONET_POSITIVE_TEST_ANNO_FILENAME = 'pos_test_48.txt'
ONET_NEGATIVE_TEST_ANNO_FILENAME = "neg_test_48.txt"
ONET_PART_TEST_ANNO_FILENAME = "part_test_48.txt"
ONET_LANDMARK_TEST_ANNO_FILENAME = "landmark_test_48.txt"

ONET_POSITIVE_VALID_ANNO_FILENAME = 'pos_valid_48.txt'
ONET_NEGATIVE_VALID_ANNO_FILENAME = "neg_valid_48.txt"
ONET_PART_VALID_ANNO_FILENAME = "part_valid_48.txt"
ONET_LANDMARK_VALID_ANNO_FILENAME = "landmark_valid_48.txt"

PNET_TRAIN_IMGLIST_FILENAME = "imglist_anno_12.txt"
RNET_TRAIN_IMGLIST_FILENAME = "imglist_anno_24.txt"
ONET_TRAIN_IMGLIST_FILENAME = "imglist_anno_48.txt"

PNET_TRAIN_IMGLIST_FILENAME_TEST = "imglist_anno_12_test.txt"
PNET_TRAIN_IMGLIST_FILENAME_VALID = "imglist_anno_12_valid.txt"

RNET_TRAIN_IMGLIST_FILENAME_TEST = "imglist_anno_24_test.txt"
RNET_TRAIN_IMGLIST_FILENAME_VALID = "imglist_anno_24_valid.txt"

ONET_TRAIN_IMGLIST_FILENAME_TEST = "imglist_anno_48_test.txt"
ONET_TRAIN_IMGLIST_FILENAME_VALID = "imglist_anno_48_valid.txt"
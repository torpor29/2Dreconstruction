import cv2
from detectFacePnet import creat_pnet, PnetDetector
from detectFaceRnet import creat_prnet, PRnetDetector
from detectFaceOnet import creat_pronet, PROnetDetector
from utils import vis_face

p_model_path = '/home/zhangyuqi/projects/model/faceDetect/results/2019-06-03_12-01-37/checkpoint.pth.tar'
image_path = '/home/zhangyuqi/NewDisk/WIDER_train/images/1--Handshaking/1_Handshaking_Handshaking_1_885.jpg'
r_model_path = '/home/zhangyuqi/projects/model/faceDetect/results/2019-06-07_13-46-00RNet/checkpoint.pth.tar'
o_model_path = '/home/zhangyuqi/projects/model/faceDetect/results/2019-06-10_18-15-59ONet/checkpoint.pth.tar'

if __name__ == '__main__':

	# test pnet
	'''pnet= creat_pnet(p_model_path, 'cuda: 1')
	pnetDetector = PnetDetector(pnet=pnet,min_face_size=12)

	img = cv2.imread(image_path)
	img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#b, g, r = cv2.split(img)
	#img2 = cv2.merge([r, g, b])

	bboxs = pnetDetector.detect_pnet(img)
	# print box_align

	vis_face(img_bg,bboxs)'''

	# test rnet
	pnet, rnet, onet = creat_pronet(p_model_path, r_model_path, o_model_path, 'cuda: 1')
	prOnetDetector = PROnetDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=12)
	img = cv2.imread(image_path)
	img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# b, g, r = cv2.split(img)
	# img2 = cv2.merge([r, g, b])

	bboxs, bboxes_align = prOnetDetector.detect_face(img)
	# print box_align

	vis_face(img_bg, bboxes_align)

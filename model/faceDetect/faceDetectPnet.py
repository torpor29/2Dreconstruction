import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(model):
	if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
		nn.init.xavier_uniform(model.weight.data)
		nn.init.constant(model.bias, 0.1)

class LossFn:
	def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
		# loss function
		self.cls_factor = cls_factor
		self.box_factor = box_factor
		self.land_factor = landmark_factor
		self.loss_cls = nn.BCELoss()
		self.loss_box = nn.MSELoss()
		self.loss_landmark = nn.MSELoss()

	def cls_loss(self, gt_label, pred_label):
		pred_label = torch.squeeze(pred_label)
		gt_label = torch.squeeze(gt_label)
		mask = torch.ge(gt_label, 0)
		valid_gt_label = torch.masked_select(gt_label, mask)
		valid_pred_label = torch.masked_select(pred_label, mask) # select valid label of gt and pred
		valid_gt_label = valid_gt_label.float()
		valid_pred_label = valid_pred_label.float()
		return self.loss_cls(valid_pred_label, valid_gt_label) * self.cls_factor

	def box_loss(self, gt_label, gt_offset, pred_offset):
		pred_offset = torch.squeeze(pred_offset)
		gt_offset = torch.squeeze(gt_offset)
		gt_label = torch.squeeze(gt_label)

		# get the mask element which != 0
		unmask = torch.eq(gt_label, 0)
		mask = torch.eq(unmask, 0)
		# convert mask to dim index
		chose_index = torch.nonzero(mask.data)
		chose_index = torch.squeeze(chose_index)
		# only valid element can effect the loss
		valid_gt_offset = gt_offset[chose_index, :]
		valid_pred_offset = pred_offset[chose_index, :]
		valid_gt_offset = valid_gt_offset.float()
		valid_pred_offset = valid_pred_offset.float()
		return self.loss_box(valid_pred_offset, valid_gt_offset) * self.box_factor


class PNet(nn.Module):
	'''PNet'''

	def __init__(self, is_train = False, use_cuda = True):
		super(PNet, self).__init__()
		self.is_train = is_train
		self.use_cude = use_cuda

		# backend
		self.pre_layer = nn.Sequential(
			nn.Conv2d(3, 10, kernel_size=3, stride=1), #conv1
			nn.PReLU(), #PReLU1
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(10, 16, kernel_size=3, stride=1),
			nn.PReLU(),
			nn.Conv2d(16, 32, kernel_size=3, stride=1),
			nn.PReLU() # PReLU3
		)
		# detection
		self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
		self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)

		self.apply(weight_init)



	def forward(self, x):
		x = self.pre_layer(x)
		label = F.sigmoid(self.conv4_1(x))
		offset = self.conv4_2(x)

		if self.is_train is True:
			return label, offset
		return label, offset

class RNet(nn.Module):
	''' RNet '''

	def __init__(self,is_train=False, use_cuda=True):
		super(RNet, self).__init__()
		self.is_train = is_train
		self.use_cuda = use_cuda
		# backend
		self.pre_layer = nn.Sequential(
			nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
			nn.PReLU(),  # prelu1
			nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
			nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
			nn.PReLU(),  # prelu2
			nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
			nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
			nn.PReLU()  # prelu3

		)
		self.conv4 = nn.Linear(64*2*2, 128)  # conv4
		self.prelu4 = nn.PReLU()  # prelu4
		# detection
		self.conv5_1 = nn.Linear(128, 1)
		# bounding box regression
		self.conv5_2 = nn.Linear(128, 4)
		# lanbmark localization
		self.conv5_3 = nn.Linear(128, 10)
		# weight initiation weih xavier
		self.apply(weight_init)

	def forward(self, x):
		# backend
		x = self.pre_layer(x)
		x = x.view(x.size(0), -1)
		x = self.conv4(x)
		x = self.prelu4(x)
		# detection
		det = torch.sigmoid(self.conv5_1(x))
		box = self.conv5_2(x)
		# landmark = self.conv5_3(x)

		if self.is_train is True:
			return det, box
		#landmard = self.conv5_3(x)
		return det, box

class ONet(nn.Module):
	''' ONet '''

	def __init__(self, is_train=False, use_cuda=True):
		super(ONet, self).__init__()
		self.is_train = is_train
		self.use_cuda = use_cuda
		# backend
		self.pre_layer = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
			nn.PReLU(),  # prelu1
			nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
			nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
			nn.PReLU(),  # prelu2
			nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
			nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
			nn.PReLU(),  # prelu3
			nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
			nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
			nn.PReLU()  # prelu4
		)
		self.conv5 = nn.Linear(128 * 2 * 2, 256)  # conv5
		self.prelu5 = nn.PReLU()  # prelu5
		# detection
		self.conv6_1 = nn.Linear(256, 1)
		# bounding box regression
		self.conv6_2 = nn.Linear(256, 4)
		# lanbmark localization
		self.conv6_3 = nn.Linear(256, 10)
		# weight initiation weih xavier
		self.apply(weight_init)

	def forward(self, x):
		# backend
		x = self.pre_layer(x)
		x = x.view(x.size(0), -1)
		x = self.conv5(x)
		x = self.prelu5(x)
		# detection
		det = torch.sigmoid(self.conv6_1(x))
		box = self.conv6_2(x)
		#landmark = self.conv6_3(x)
		if self.is_train is True:
			return det, box
		# landmard = self.conv5_3(x)
		return det, box


# Residual Block
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.downsample = downsample

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.downsample:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)
		return out

# Residual Module
class ResNet(nn.Module):
	def __init__(self, block, num_classes=10):
		super(ResNet, self).__init__()
		self.in_channels = 16
		self.conv = nn.Conv2d(3, 16, kernel_size=3)
		self.bn = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)
		self.layer1 = self.make_layer(block, 16, 3)
		self.layer2 = self.make_layer(block, 32, 3, 2)
		self.layer3 = self.make_layer(block, 64, 3, 2)
		self.avg_pool = nn.AvgPool2d(8)
		self.fc = nn.Linear(64, num_classes)

	def make_layer(self, block, out_channels, blocks, stride=1):
		downsample = None
		if (stride != 1) or (self.in_channels != out_channels):
			downsample = nn.Sequential(
				nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride),
				nn.BatchNorm2d(out_channels))
		layers = []
		layers.append(block(self.in_channels, out_channels, stride, downsample))
		self.in_channels = out_channels
		for i in range(1, blocks):
			layers.append(block(out_channels, out_channels))
		return  nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv(x)
		out = self.bn(out)
		out = self.relu(out)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.avg_pool(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out
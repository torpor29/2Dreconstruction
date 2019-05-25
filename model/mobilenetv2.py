from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import sys
# sys.path.append("..")
# from utils.utils import MyDataset,validate, show_confMat
# from tensorboardX import SummaryWriter
# from datetime import datetime
# sys.path.append("/home/zhangyuqi/clone/facedetect")
# train_txt_path=''
# valid_txt_path=''
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
# -----------------------------define network----------------------------------------
class Bottleneck(nn.Module):

	# Bottleneck implementation
	def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU):
		super(Bottleneck,self).__init__()
		self.conv1=nn.Conv2d(inplanes, inplanes*t, kernel_size=1, bias=False)
		self.bn1=nn.BatchNorm2d(inplanes*t)
		self.conv2=nn.Conv2d(inplanes*t, inplanes*t, kernel_size=3, stride=stride,
							 padding=1, bias=False, groups=inplanes*t)
		self.bn2=nn.BatchNorm2d(inplanes*t)
		self.conv3=nn.Conv2d(inplanes*t, outplanes, kernel_size=1, bias=False)
		self.bn3=nn.BatchNorm2d(outplanes)
		self.activation=activation(inplace=True)
		self.stride=stride
		self.t=t
		self.inplanes=inplanes
		self.outplanes=outplanes

	def forward(self, x):
		residual =x

		out=self.conv1(x)
		out=self.bn1(out)
		out=self.activation(out)

		out=self.conv2(out)
		out=self.bn2(out)
		out=self.activation(out)

		out=self.conv3(out)
		out=self.bn3(out)

		if self.stride==1 and self.inplanes==self.outplanes:
			out += residual

		return out

class MobileNetV2(nn.Module):


	# MobileNetV2 implementation

	def __init__(self, scale=1.0, input_size=48, t=6, in_channels=3, num_classes=10, activation=nn.ReLU6):
		'''MobileNetV2 constructor.
		:param in_channels:(int, optional): number of channels in the input tensor.
			Default is 3 for RGB image inputs.
		:param input_size:
		:param num_classes: number of classes to predict. Default is number of labels.
		:param scale:
		:param t:
		:param activation
		'''
		# ---------------------conv2d 48*48*3->24*24*16------------------------
		super(MobileNetV2, self).__init__()
		self.scale=scale
		self.t=t
		self.activation_type=activation
		self.activation=activation(inplace=True)
		self.num_classes=num_classes
		self.num_of_channels=[16, 24, 32, 64, 256, 10]
		self.c=self.num_of_channels
		self.n=[1,2,2,2]
		self.s=[2,2,2,2]

		self.conv1=nn.Conv2d(in_channels,self.c[0] , kernel_size=3, bias=False, stride=self.s[0], padding=1)
		self.bn1=nn.BatchNorm2d(self.c[0])

		# --------------bottleneck_1 24*24*16->12*12*24->6*6*32->3*3*64----------
		self.bottlenecks = self._make_bottleneck()

		# --------------fc-------------------------------------------------------
		self.fc1=nn.Linear(self.c[3] * 100, self.c[4])
		self.dp1=nn.Dropout(0.5)
		self.fc2=nn.Linear(self.c[4],self.c[5])
		self.dp2=nn.Dropout()


		self.init_params()

	def init_params(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal(m.weight, mode='fan_out')
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				init.constant_(m.weight, 1)
				init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				init.normal_(m.weight, std=0.01)
				if m.bias is not None:
					init.constant_(m.bias, 0)


	def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
		modules=OrderedDict()
		stage_name="LinearBottleneck{}".format(stage)

		# first module is the only one utilizing stride

		first_module=Bottleneck(inplanes=inplanes, outplanes=outplanes, stride=stride, t=t,
								activation=self.activation_type)
		modules[stage_name+"_0"]=first_module

		# add more bottlenecks
		for i in range(n-1):
			name=stage_name+"_{}".format(i+1)
			module=Bottleneck(inplanes=outplanes, outplanes=outplanes, stride=1, t=6,
							  activation=self.activation_type)
			modules[name]=module
		return nn.Sequential(modules)


	def _make_bottleneck(self):
		modules=OrderedDict()
		stage_name="Bottlenecks"

		# First module is the only one with t=1

		bottleneck1 = self._make_stage(inplanes=self.c[0], outplanes=self.c[1], n=self.n[1], stride=self.s[1],
									 t=self.t,stage=0)
		modules[stage_name+"_0"]=bottleneck1

		# add more Bottlenecks
		for i in range(1,len(self.c) - 3):
			name=stage_name+"_{}".format(i)
			module=self._make_stage(inplanes=self.c[i], outplanes=self.c[i+1], n=self.n[i+1],
									stride=self.s[i+1], t=self.t, stage=i)
			modules[name]=module
		return nn.Sequential(modules)



	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.activation(x)

		x = self.bottlenecks(x)
		x = self.activation(x)

		x = x.view(x.size(0), -1)

		x = self.fc1(x)
		x = self.activation(x)
		x = self.dp1(x)
		x = self.fc2(x)



		return x

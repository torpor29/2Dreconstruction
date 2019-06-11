import torch
import torch.nn as nn
from torch.nn import init

class miniVGG(nn.Module):
	def __init__(self, inchannels=3, activation=nn.ReLU6):
		super(miniVGG, self).__init__()
		self.inchannels = inchannels
		self.activation = activation(inplace=True)
		self.conv1 = nn.Conv2d(inchannels, 96, kernel_size=3, stride=2)
		self.conv1b = nn.Conv2d(96, 96, kernel_size=3, stride=3)
		self.norm = nn.LocalResponseNorm(96, 0.0005, 0.75, 2)
		self.conv2 = nn.Conv2d(96, 256, kernel_size=3, stride=1)
		self.conv2b = nn.Conv2d(256, 256, kernel_size=3, stride=3)
		self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
		self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
		self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2)
		self.fc7 = nn.Linear(1, 2048)
		self.points = nn.Linear(2048, 136)
		self.dp = nn.Dropout(0.5)
		self.init_params()

	def init_params(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.xavier_uniform(m.weight.data)
				init.constant_(m.bias, 0.1)
			elif isinstance(m, nn.Linear):
				init.xavier_uniform(m.weight.data)
				init.constant_(m.bias, 0.1)


	def forward(self, x):
		x = self.conv1(x)
		x = self.conv1b(x)
		x = self.activation(x)
		x = self.norm(x)
		x = self.conv2(x)
		x = self.conv2b(x)
		x = self.activation(x)
		x = self.conv3(x)
		x = self.activation(x)
		x = self.conv4(x)
		x = self.activation(x)
		x = self.conv5(x)
		x = self.activation(x)
		x = self.fc7(x)
		x = self.activation(x)
		x = self.dp(x)
		x = self.points(x)
		return x

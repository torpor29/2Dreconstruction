from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class MyDataset(Dataset):
	def __init__(self, txt_path, transform=None, target_transform=None):
		fh = open(txt_path, 'r')
		imgs = []
		for line in fh:
			line = line.rstrip()
			words = line.split()
			# imgs.append(words[0])
			la = []
			for i in range(1, len(words)):
				la.append((float(words[i]) - 80) / 80)
			imgs.append((words[0], la))

		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform


	def __getitem__(self, index):
		fn, label = self.imgs[index]
		img = Image.open(fn).convert('RGB')

		label = torch.Tensor(label)



		if self.transform is not None:
			img = self.transform(img)

		return fn, img, label

	def __len__(self):
		return len(self.imgs)

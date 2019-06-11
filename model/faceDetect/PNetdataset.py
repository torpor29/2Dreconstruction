from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import  torchvision.transforms as transforms
trainsform = transforms.ToTensor()
class MyDataset(Dataset):
	def __init__(self, imdb):
		'''fh = open(txt_path, 'r')
		imgs = []
		for line in fh:
			line = line.rstrip()
			words = line.split(' ')
			# imgs.append(words[0])
			la = []
			print(words[0])
			for i in range(1, len(words)):
				la.append((float(words[i]) - 80) / 80)
			imgs.append((words[0], la))

		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform'''

		self.imdb = imdb


	def __getitem__(self, index):
		im = cv2.imread(self.imdb[index]['image'])
		if self.imdb[index]['flipped']:
			im = im[:, ::-1, :]

		cls = self.imdb[index]['label']
		bbox_target = self.imdb[index]['bbox_target']
		imcopy = im.copy()
		imcopy = imcopy.astype(np.float)


		#fn, label = self.imgs[index]
		#img = Image.open(fn).convert('RGB')

		#label = torch.Tensor(label)



		#if self.transform is not None:
			#img = self.transform(img)

		return self.imdb[index]['image'], trainsform(imcopy), cls, bbox_target

	def __len__(self):
		return len(self.imdb)

def get_loaders(imdb, train_batch_size, workers):
	#val_data = MyDataset(imdb)
	#val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=False,
											# num_workers=workers, pin_memory=True)

	train_data = MyDataset(imdb)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
											   num_workers=workers, pin_memory=True)

	return train_loader
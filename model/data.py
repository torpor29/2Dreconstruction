import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms
from mydataset import MyDataset
import sys
sys.path.append('/home/zhangyuqi/clone/facedetect')

_imagenet_stats={'mean': [0.502, 0.424, 0.380],
				 'std': [0.308, 0.289, 0.287]}

def inception_preprocess(input_size, normalize=_imagenet_stats):
	return transforms.Compose([
		transforms.RandomResizedCrop(input_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(**normalize)
	])

def scale_crop(input_size, scale_size=None, normalize=_imagenet_stats):
	t_list = [
		transforms.CenterCrop(input_size),
		transforms.ToTensor(),
		transforms.Normalize(**normalize),
	]
	if scale_size != input_size:
		t_list = [transforms.Resize(scale_size)] + t_list
	return transforms.Compose(t_list)

def get_transform(augment = True, input_size = 160):
	normalize = _imagenet_stats
	scale_size = int(input_size / 0.875)
	if augment:
		return inception_preprocess(input_size=input_size, normalize=normalize)
	else:
		return scale_crop(input_size=input_size, scale_size=scale_size, normalize=normalize)
	# resize the image, then labels should be changed relatively

def get_loaders(dataroot, val_batch_size, train_batch_size, input_size, workers):
	val_data = MyDataset(txt_path=os.path.join(dataroot,'raw_valid_label.txt'),
									transform=get_transform(False, input_size))
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=False,
											 num_workers=workers, pin_memory=True)

	train_data = MyDataset(txt_path=os.path.join(dataroot, 'raw_train_label.txt'),
									  transform=get_transform(input_size=input_size))
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True,
											   num_workers=workers, pin_memory=True)

	return train_loader, val_loader

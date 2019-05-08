import os
import shutil

import matplotlib
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm, trange

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from cycliclr import CyclicLR

def train(model, loader, epoch, optimizer, criterion, device, dtype, batch_size, log_interval, scheduler):
	model.train()
	correct1, correct5 = 0, 0

	for batch_idx, (path, data, target) in enumerate(tqdm(loader)):
		if isinstance(scheduler, CyclicLR):
			scheduler.batch_step()
		data, target = data.to(device=device, dtype=dtype), target.to(device=device)

		optimizer.zero_grad()
		output = model(data)

		loss = criterion(output, target)

		loss.backward()
		optimizer.step()

		#corr = correct(output, target, topk=(1, 5))
		#correct1 += corr[0]
		#correct5 += corr[1]

		if batch_idx % log_interval == 0:
			tqdm.write(
				'Train Epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}. '.format(epoch, batch_idx, len(loader),
															100. * batch_idx / len(loader), loss.item()))
	return loss.item(), correct1 / len(loader.dataset), correct5 / len(loader.dataset)

def test(save_path, model, loader, criterion, device, dtype):
	model.eval()
	test_loss = 0
	correct1, correct5 = 0, 0
	path = save_path + '/test_label.txt'
	f = open(path, 'w')
	for batch_idx, (path, data, target) in enumerate(tqdm(loader)):
		data, target = data.to(device=device, dtype=dtype), target.to(device=device)
		with torch.no_grad():
			output = model(data)
			test_loss += criterion(output, target).item()
			outputnp = output.cpu().numpy()
			for i in range(0, len(path)):
				newline = path[i]
				for j in range(0, 10):
					newline = newline + ' ' + str(outputnp[i][j])
				f.write(newline+'\n')
	f.close()

			#corr = correct(output, target, topk=(1,5))
		#correct1 += corr[0]
		#correct5 += corr[1]

	test_loss /= len(loader)

	tqdm.write(
		'\nTest set: Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
		'Top5: {}/{} ({:.2f}%). '.format(test_loss, int(correct1), len(loader.dataset),
										 100. * correct1 / len(loader.dataset), int(correct5),
										len(loader.dataset), 100. * correct5 / len(loader.dataset)))
	return test_loss, correct1 / len(loader.dataset), correct5 / len(loader.dataset)

def correct(output, target, topk=(1,)):
	"""maxk = max(topk)

	_, pred = output.topk(maxk, 1,True, True)
	pred = pred.t().type_as(target)
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0).item()
		res.append(correct_k)"""
	outputnp = output.numpy()
	targetnp = target.cpu().numpy()
	res = outputnp - targetnp
	res = res.abs().sum()
	return res

def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar'):
	save_path = os.path.join(filepath, filename)
	best_path = os.path.join(filepath, 'model_best.pth.tar')
	torch.save(state, save_path)
	if is_best:
		shutil.copyfile(save_path, best_path)

def find_bounds_clr(model, loader, optimizer, criterion, device, dtype, min_lr=8e-6,
					max_lr=8e-5, step_size=2000, mode='triangular', save_path='./'):
	model.train()
	correct1, correct5 = 0, 0
	scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size=step_size, mode=mode)
	epoch_count = step_size // len(loader)
	accuracy = []

	for _ in trange(epoch_count):
		for batch_idx, (path, data, target) in enumerate(tqdm(loader)):
			if scheduler is not None:
				scheduler.batch_step()
			data, target = data.to(device=device, dtype=dtype), target.to(device=device)

			optimizer.zero_grad()
			output = model(data)

			loss = criterion(output, target)

			loss.backward()
			optimizer.step()


			#corr = correct(output, target)
			accuracy.append(loss / data.shape[0])

	lrs = np.linspace(min_lr, max_lr, step_size)
	plt.plot(lrs, accuracy)
	plt.savefig(os.path.join(save_path, 'acc.jpg'))
	plt.show()

	return


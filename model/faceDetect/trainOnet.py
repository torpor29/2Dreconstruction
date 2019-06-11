import argparse
import sys
import os
import config
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime
from faceDetectPnet import ONet, LossFn

from PNetdataset import get_loaders
from ImageDB import ImageDB
from torch.optim.lr_scheduler import MultiStepLR

from loggerPnet import CsvLogger
import csv
from tqdm import tqdm, trange
import shutil


def parse_Args():
	parser = argparse.ArgumentParser(description='Train ONet',
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--anno_file', dest='anno_file',
	                    default=os.path.join(config.ANNO_STORE_DIR, config.ONET_TRAIN_IMGLIST_FILENAME),
	                    help='training data annotation file', type=str)
	parser.add_argument('--model', dest='model_store_path', help='training model store dir',
	                    default=config.MODEL_STORE_DIR, type=str)
	parser.add_argument('--epoch', dest='end_epoch', help='end epoch of training',
	                    default=config.END_EPOCH, type=int)
	parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
	                    default=200, type=int)
	parser.add_argument('--lr', dest='lr', help='learning rate',
	                    default=config.TRAIN_LR, type=float)
	parser.add_argument('--batch_size', dest='batch_size', help='train batch size',
	                    default=config.TRAIN_BATCH_SIZE, type=int)
	parser.add_argument('--gpu', dest='use_cuda', help='train with gpu',
	                    default=config.USE_CUDA, type=bool)
	parser.add_argument('--prefix_path', dest='prefix_path', help='training data annotation images prefix root path',
	                    type=str)
	parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')
	parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
	parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
	parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
	                    help='path to latest checkpoint (default: none)')
	parser.add_argument('--gpus', default=None, help='List of GPUs used for training - e.g 0,1,3')
	parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')
	parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
	                    help='manual epoch number (useful on restarts)')
	parser.add_argument('--log_interval', type=int, default=100, metavar='N',
	                    help='Number of batches between log messages')
	parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
	                    help='Number of data loading workers (default: 4)')

	args = parser.parse_args()
	return args

def main():
	args = parse_Args()
	if args.seed is None:
		args.seed = random.randint(1, 10000)
	print("Random Seed: ", args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.gpus:
		torch.cuda.manual_seed_all(args.seed)


	time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	if args.evaluate:
		args.results_dir = './tmpPNet'

	save_path = os.path.join(args.results_dir, time_stamp + 'ONet')
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	if args.gpus is not None:
		args.gpus = [int(i) for i in args.gpus.split(',')]
		device = 'cuda:' + str(args.gpus[0])
		cudnn.benchmark = True
	else:
		device = 'cpu'

	if args.type == 'float64':
		dtype = torch.float64
	elif args.type == 'float32':
		dtype = torch.float32
	elif args.type == 'float16':
		dtype = torch.float16
	else:
		raise ValueError('Wrong type!')  # TODO int8

	model = ONet(is_train=True, use_cuda=True)
	if args.gpus is not None:
		model = torch.nn.DataParallel(model, args.gpus)
	model.to(device=device, dtype=dtype)


	num_parameters = sum([l.nelement() for l in model.parameters()])
	print(model)
	print('number of parameters: {}'.format(num_parameters))

	lossfn = LossFn()
	lossfn.loss_cls.to(device=device, dtype=dtype)
	lossfn.loss_box.to(device=device, dtype=dtype)

	imagedb = ImageDB(args.anno_file, args.prefix_path, 'train')
	imagedb_valid = ImageDB(os.path.join(config.ANNO_STORE_DIR,
	                                     config.ONET_TRAIN_IMGLIST_FILENAME_VALID), 'train')
	imdb = imagedb.load_imdb()
	imdb_val = imagedb_valid.load_imdb()
	train_loader = get_loaders(imdb, args.batch_size, args.workers)
	valid_loader = get_loaders(imdb_val, args.batch_size, args.workers)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = MultiStepLR(optimizer, milestones=[200, 300], gamma=0.1)

	best_test = 0

	# optionally resume from a checkpoint
	data = None
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume, map_location=device)
			args.start_epoch = checkpoint['epoch'] - 1
			best_test = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
			      .format(args.resume, checkpoint['epoch']))
		elif os.path.isdir(args.resume):
			checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')
			csv_path = os.path.join(args.resume, 'results.csv')
			print("=> loading checkpoint '{}'".format(checkpoint_path))
			checkpoint = torch.load(checkpoint_path, map_location=device)
			args.start_epoch = checkpoint['epoch'] - 1
			best_test = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
			data = []
			with open(csv_path) as csvfile:
				reader = csv.DictReader(csvfile)
				for row in reader:
					data.append(row)
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	'''if args.evaluate:
		loss, top1, top5 = test(save_path, model, val_loader, criterion, device, dtype)  # TODO
		return'''

	csv_logger = CsvLogger(filepath=save_path, data=data)
	csv_logger.save_params(sys.argv, args)

	train_network(args.start_epoch, args.end_epoch, scheduler, model, train_loader, valid_loader, optimizer, lossfn,
	              device, dtype, args.batch_size, args.log_interval, csv_logger, save_path, best_test)

def train_network(start_epoch, epochs, scheduler, model, train_loader, valid_loader, optimizer, lossfn, device, dtype,
                  batch_size, log_interval, csv_logger, save_path, best_test):
	for epoch in trange(start_epoch, epochs + 1):
		'''if not isinstance(scheduler, CyclicLR):
			scheduler.step()'''
		train_loss, acc, cls_loss, bbox_loss = train(model, train_loader, epoch, optimizer, lossfn,
		                                             device, dtype, batch_size, log_interval, scheduler)
		test_loss = test(save_path, model, valid_loader, lossfn, device, dtype)

		if epoch > 1:
			csv_logger.write({'epoch': epoch + 1, 'accuracy': acc, 'cls_loss': cls_loss,
			                  'bbox_loss': bbox_loss, 'test_loss': test_loss})
			save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
			                 'optimizer': optimizer.state_dict()}, acc > best_test, filepath=save_path)

			csv_logger.plot_progress()

		if acc > best_test:
			best_test = acc

	csv_logger.write_text('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))

def train(model, train_loader, epoch, optimizer, lossfn, device,
          dtype, batch_size, log_interval, scheduler):

	model.train()
	correct1, correct5 = 0, 0
	accuracy_list = []
	cls_loss_list = []
	bbox_loss_list = []

	for batch_idx, (path, data, cls, bbox) in enumerate(tqdm(train_loader)):
		'''if isinstance(scheduler, CyclicLR):
			scheduler.batch_step()'''
		data, cls, bbox = data.to(device=device, dtype=dtype), cls.to(device=device, dtype=dtype), bbox.to(device=device, dtype=dtype)

		optimizer.zero_grad()
		'''data = Variable(data)
		cls = Variable(cls)
		bbox = Variable(bbox)'''


		outputcls, outputbbox = model(data)

		#loss = criterion(output, target)

		cls_loss = lossfn.cls_loss(cls, outputcls)
		box_offset_loss = lossfn.box_loss(cls, bbox, outputbbox)
		all_loss = cls_loss * 1.0 + box_offset_loss * 0.5

		all_loss.backward()
		optimizer.step()

		# corr = correct(output, target, topk=(1, 5))
		# correct1 += corr[0]
		# correct5 += corr[1]

		if batch_idx % log_interval == 0:
			accuracy = compute_accuracy(outputcls, cls)

			show1 = accuracy.data.tolist()
			show2 = cls_loss.data.tolist()
			show3 = box_offset_loss.data.tolist()
			show5 = all_loss.data.tolist()

			tqdm.write(
				'Train Epoch: {} [{}/{} ({:.0f}%]\taccuracy: {},clsLoss: {:.6f}, bbox loss: {},'
				'all_loss: {} '.format(epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader),
				                       show1, show2, show3, show5))

			accuracy_list.append(accuracy)
			cls_loss_list.append(cls_loss)
			bbox_loss_list.append(box_offset_loss)

	accuracy_avg = torch.mean(torch.tensor(accuracy_list))
	cls_loss_avg = torch.mean(torch.tensor(cls_loss_list))
	bbox_loss_avg = torch.mean(torch.tensor(bbox_loss_list))
	# landmark_loss_avg = torch.mean(torch.cat(landmark_loss_list))

	show6 = accuracy_avg.data.tolist()
	show7 = cls_loss_avg.data.tolist()
	show8 = bbox_loss_avg.data.tolist()

	return all_loss.item(), show6, show7, show8

def test(save_path, model, loader, lossfn, device, dtype):
	model.eval()
	test_loss = 0
	correct1, correct5 = 0, 0
	file_path = save_path + '/test_label.txt'
	for batch_idx, (path, data, cls, bbox) in enumerate(tqdm(loader)):
		'''if isinstance(scheduler, CyclicLR):
			scheduler.batch_step()'''
		data, cls, bbox = data.to(device=device, dtype=dtype), cls.to(device=device, dtype=dtype), bbox.to(device=device, dtype=dtype)
		with torch.no_grad():
			outputcls, outputbbox = model(data)
			cls_loss = lossfn.cls_loss(cls, outputcls)
			box_offset_loss = lossfn.box_loss(cls, bbox, outputbbox)
			all_loss = cls_loss * 1.0 + box_offset_loss * 0.5
			test_loss += all_loss.item()

	test_loss /= len(loader)

	tqdm.write(
		'\nTest set: Average loss: {:.4f}'.format(test_loss))
	return test_loss

def compute_accuracy(prob_cls, gt_cls):

	prob_cls = torch.squeeze(prob_cls)
	gt_cls = torch.squeeze(gt_cls)

	#we only need the detection which >= 0
	mask = torch.ge(gt_cls,0)
	#get valid element
	valid_gt_cls = torch.masked_select(gt_cls,mask)
	valid_prob_cls = torch.masked_select(prob_cls,mask)
	size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
	prob_ones = torch.ge(valid_prob_cls,0.6).float()
	right_ones = torch.eq(prob_ones,valid_gt_cls).float()

	return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))

def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar'):
	save_path = os.path.join(filepath, filename)
	best_path = os.path.join(filepath, 'model_best.pth.tar')
	torch.save(state, save_path)
	if is_best:
		shutil.copyfile(save_path, best_path)

if __name__ == '__main__':
	main()

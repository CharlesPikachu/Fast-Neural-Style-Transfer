'''
Function:
	train the model
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import cfg
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from modules.utils import *
from torchvision import datasets
from torch.utils.data import DataLoader
from modules.nets import VGG16, TransformerNet


'''train'''
def train(cfg):
	# prepare
	logger = Logger(cfg.logfilepath)
	checkpoint_dir = os.path.join(cfg.backupdir, 'checkpoint')
	rendering_dir = os.path.join(cfg.backupdir, 'rendering')
	checkDir(cfg.backupdir)
	checkDir(checkpoint_dir)
	checkDir(rendering_dir)
	use_cuda = torch.cuda.is_available()
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	# dataset
	train_dataset = datasets.ImageFolder(cfg.datasetpath, transformData(cfg.image_size, image_type='train', mean=cfg.mean, std=cfg.std))
	dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
	# model
	transformer_net = TransformerNet()
	vgg_net = VGG16(pretrained=True, requires_grad=False)
	if use_cuda:
		transformer_net = transformer_net.cuda()
		vgg_net = vgg_net.cuda()
	# restore
	if cfg.checkpointpath_restore:
		transformer_net.load_state_dict(torch.load(cfg.checkpointpath_restore))
	# optimizer
	optimizer = optim.Adam(transformer_net.parameters(), cfg.learning_rate)
	# loss function
	l2_loss = nn.MSELoss().cuda() if use_cuda else nn.MSELoss()
	# extract style features
	image_style = loadStyleImage(cfg.style_image_path, transformData(cfg.image_size, image_type='style', mean=cfg.mean, std=cfg.std))
	image_style = image_style.repeat(cfg.batch_size, 1, 1, 1).type(FloatTensor)
	features_style = vgg_net(image_style)
	gram_style = [getGramMatrix(x) for x in features_style]
	# train
	for epoch in range(cfg.num_epochs):
		epoch_metrics = {'content_loss': [], 'style_loss': [], 'total_loss': []}
		for batch_i, (images, _) in enumerate(dataloader):
			optimizer.zero_grad()
			images_ori = images.type(FloatTensor)
			images_transform = transformer_net(images_ori)
			# --extract features
			features_ori = vgg_net(images_ori)
			features_transform = vgg_net(images_transform)
			# --content loss
			content_loss = l2_loss(features_ori[1], features_transform[1]) * 1e5
			# --style loss
			style_loss = 0
			for x1, x2 in zip(features_transform, gram_style):
				style_loss += l2_loss(getGramMatrix(x1), x2[:images.size(0), :, :])
			style_loss = style_loss * 1e10
			total_loss = content_loss + style_loss
			total_loss.backward()
			optimizer.step()
			epoch_metrics['content_loss'] += [content_loss.item()]
			epoch_metrics['style_loss'] += [style_loss.item()]
			epoch_metrics['total_loss'] += [total_loss.item()]
			# --logging
			logger.info('[Epoch]: %d/%d, [Batch]: %d/%d, [Content_Loss]: %.2f (%.2f), [Style_Loss]: %.2f (%.2f), [Total_loss]: %.2f (%.2f)...'
						% (epoch+1, cfg.num_epochs, batch_i, len(dataloader), content_loss.item(), np.mean(epoch_metrics.get('content_loss')), style_loss.item(), np.mean(epoch_metrics.get('style_loss')), total_loss.item(), np.mean(epoch_metrics.get('total_loss'))))
			# --save model
			num_batches = epoch * len(dataloader) + batch_i + 1
			if num_batches % cfg.save_interval == 0 or (batch_i + 1) == len(dataloader):
				style_name = cfg.style_image_path.split('/')[-1].split('.')[0]
				saveSamplesBatch(images_ori, mean=cfg.mean, std=cfg.std, transformer_net=transformer_net, savepath=os.path.join(rendering_dir, '%s_%s.jpg' % (style_name, num_batches)))
				torch.save(transformer_net.state_dict(), os.path.join(checkpoint_dir, '%s_%s.pth' % (style_name, num_batches)))


'''run'''
if __name__ == '__main__':
	train(cfg)
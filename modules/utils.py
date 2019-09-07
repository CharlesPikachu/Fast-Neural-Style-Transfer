'''
Function:
	some utils
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import av
import torch
import logging
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image


'''transform image'''
def transformData(image_size, image_type='train', mean=None, std=None):
	if image_type == 'train':
		transform = transforms.Compose([transforms.Resize((int(image_size[0]*1.2), int(image_size[1]*1.2))),
										transforms.RandomCrop(image_size),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])
	elif image_type == 'style':
		transform = transforms.Compose([transforms.Resize(image_size),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])
	else:
		raise ValueError('transformData.image_type unsupport <%s>...' % image_type)
	return transform


'''gram matrix'''
def getGramMatrix(features):
	batch_size, num_channels, h, w = features.size()
	m = features.view(batch_size, num_channels, w*h)
	m_t = m.transpose(1, 2)
	gram = m.bmm(m_t) / (num_channels * h * w)
	return gram


'''reconstruct image from torch.tensor'''
def reconstructImage(image_tensor, mean, std):
	for c in range(3):
		image_tensor[:, c].mul_(std[c]).add_(mean[c])
	image_tensor *= 255
	image = torch.clamp(image_tensor, 0, 255).cpu().numpy().astype(np.uint8)
	image = image.transpose(1, 2, 0)
	return image


'''check the existence of dir'''
def checkDir(dirname):
	if not os.path.exists(dirname):
		os.mkdir(dirname)


'''save samples while training'''
def saveSamplesBatch(images_batch, mean, std, transformer_net, savepath):
	transformer_net.eval()
	with torch.no_grad():
		output = transformer_net(images_batch)
	images_grid = torch.cat((images_batch.cpu(), output.cpu()), 2)
	for c in range(3):
		images_grid[:, c].mul_(std[c]).add_(mean[c])
	save_image(images_grid, savepath, nrow=4)
	transformer_net.train()


'''load style image'''
def loadStyleImage(style_image_path, transform):
	img = Image.open(style_image_path)
	return transform(img)


'''logger'''
class Logger():
	def __init__(self, logfilepath, **kwargs):
		logging.basicConfig(level=logging.INFO,
							format='%(asctime)s %(levelname)-8s %(message)s',
							datefmt='%Y-%m-%d %H:%M:%S',
							handlers=[logging.FileHandler(logfilepath),
									  logging.StreamHandler()])
	@staticmethod
	def log(level, message):
		logging.log(level, message)
	@staticmethod
	def debug(message):
		Logger.log(logging.DEBUG, message)
	@staticmethod
	def info(message):
		Logger.log(logging.INFO, message)
	@staticmethod
	def warning(message):
		Logger.log(logging.WARNING, message)
	@staticmethod
	def error(message):
		Logger.log(logging.ERROR, message)


'''extract frames'''
def extractFrames(videopath):
	video = av.open(videopath)
	for frame in video.decode(0):
		yield frame.to_image()
'''
Function:
	define the network models
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


'''vgg 16'''
class VGG16(nn.Module):
	def __init__(self, pretrained=True, requires_grad=False, **kwargs):
		super(VGG16, self).__init__()
		self.slice1 = nn.Sequential()
		self.slice2 = nn.Sequential()
		self.slice3 = nn.Sequential()
		self.slice4 = nn.Sequential()
		vgg_features = models.vgg16(pretrained=pretrained).features
		for i in range(4):
			self.slice1.add_module(str(i), vgg_features[i])
		for i in range(4, 9):
			self.slice2.add_module(str(i), vgg_features[i])
		for i in range(9, 16):
			self.slice3.add_module(str(i), vgg_features[i])
		for i in range(16, 23):
			self.slice3.add_module(str(i), vgg_features[i])
		if not requires_grad:
			for parameters in self.parameters():
				parameters.requires_grad = False
	def forward(self, x):
		x1 = self.slice1(x)
		x2 = self.slice2(x1)
		x3 = self.slice3(x2)
		x4 = self.slice4(x3)
		return [x1, x2, x3, x4]


'''residual block'''
class ResidualBlock(nn.Module):
	def __init__(self, channels):
		super(ResidualBlock, self).__init__()
		self.block = nn.Sequential(BasicConv(channels, channels, kernel_size=3, stride=1, use_normalize=True, use_relu=True, use_upsample=False),
								   BasicConv(channels, channels, kernel_size=3, stride=1, use_normalize=True, use_relu=False, use_upsample=False))
	def forward(self, x):
		return self.block(x) + x


'''basic conv'''
class BasicConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_upsample=False, use_normalize=True, use_relu=True, **kwargs):
		super(BasicConv, self).__init__()
		self.block = nn.Sequential(nn.ReflectionPad2d(kernel_size // 2),
								   nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))
		self.use_upsample = use_upsample
		self.normalize = nn.InstanceNorm2d(out_channels, affine=True) if use_normalize else None
		self.use_relu = use_relu
	def forward(self, x):
		if self.use_upsample:
			x = F.interpolate(x, scale_factor=2)
		x = self.block(x)
		if self.normalize is not None:
			x = self.normalize(x)
		if self.use_relu:
			x = F.relu(x, inplace=True)
		return x


'''transformer net'''
class TransformerNet(nn.Module):
	def __init__(self):
		super(TransformerNet, self).__init__()
		self.network = nn.Sequential(BasicConv(3, 32, kernel_size=9, stride=1),
									 BasicConv(32, 64, kernel_size=3, stride=2),
									 BasicConv(64, 128, kernel_size=3, stride=2),
									 ResidualBlock(128),
									 ResidualBlock(128),
									 ResidualBlock(128),
									 ResidualBlock(128),
									 ResidualBlock(128),
									 BasicConv(128, 64, kernel_size=3, use_upsample=True),
									 BasicConv(64, 32, kernel_size=3, use_upsample=True),
									 BasicConv(32, 3, kernel_size=9, use_upsample=False, use_normalize=False, use_relu=False))
	def forward(self, x):
		return self.network(x)
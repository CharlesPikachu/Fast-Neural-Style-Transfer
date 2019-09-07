'''
Function:
	test the model
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import cfg
import time
import tqdm
import torch
import argparse
import skvideo.io
import torchvision.transforms as transforms
from PIL import Image
from modules.utils import *
from modules.nets import TransformerNet
from torchvision.utils import save_image


'''test'''
def test(datapath, checkpointpath, outputpath):
	# prepare
	use_cuda = torch.cuda.is_available()
	checkDir(outputpath)
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	# model
	transformer_net = TransformerNet()
	if use_cuda:
		transformer_net = transformer_net.cuda()
	transformer_net.load_state_dict(torch.load(checkpointpath))
	transformer_net.eval()
	# transform
	transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize(cfg.mean, cfg.std)])
	# test model
	ext = datapath.split('.')[-1]
	# --image
	if ext.lower() in ['png', 'jpg', 'bmp', 'jpeg']:
		image = Image.open(datapath)
		image = transform(image).type(FloatTensor)
		image = image.unsqueeze(0)
		with torch.no_grad():
			image_transformer = transformer_net(image).cpu()
		for c in range(3):
			image_transformer[:, c].mul_(cfg.std[c]).add_(cfg.mean[c])
		save_image(image_transformer, os.path.join(outputpath, 'output_%d.%s' % (int(time.time()), ext)))
	# --video
	elif ext.lower() in ['avi', 'mp4']:
		frames = []
		for frame in tqdm.tqdm(extractFrames(datapath), desc='Process video'):
			with torch.no_grad():
				frames += [reconstructImage(transformer_net(transform(frame).unsqueeze(0))[0])]
		writer = skvideo.io.FFmpegWriter('output_%d.gif' % int(time.time()))
		for frame in tqdm.tqdm(frames, desc='Saving result'):
			writer.writeFrame(frame)
		writer.close()
	# --unsupport
	else:
		raise ValueError('Unsupport file type --> extension <%s>...' % ext)


'''run'''
if __name__ == '__main__':
	parser = argparse.ArgumentParser('Fast Neural Style')
	parser.add_argument("--datapath", type=str, required=True, help="Path of image/video.")
	parser.add_argument("--checkpointpath", type=str, required=True, help="Path of checkpoint model.")
	parser.add_argument("--outputpath", type=str, required=True, help="Path to save results.", default='outputs')
	args = parser.parse_args()
	test(datapath=args.datapath, checkpointpath=args.checkpointpath, outputpath=args.outputpath)
'''config file'''


'''dataset'''
datasetpath = 'dataset/images' # use coco(http://cocodataset.org/#home) in original paper
					'''
					datasetpath example:
						-images
							--folders1
							--folders2
							...
							--foldersn
					'''
style_image_path = 'dataset/styles/starry_night.jpg' # the target style image path


'''train'''
num_epochs = 5 # training epochs
batch_size = 4 # batch size
image_size = (256, 256) # input image size
learning_rate = 1e-3 # learning rate
mean = [0.485, 0.456, 0.406] # used for pytorch pretrained models
std = [0.229, 0.224, 0.225] # used for pytorch pretrained models
backupdir = 'backup' # backup folder
save_interval = 500 # every save_interval batchs, save the model
logfilepath = 'train.log' # file to save the train log
checkpointpath_restore = '' # used to restore model
style_loss_weight = 5e5 # used to balance the style loss and content loss
content_loss_weight = 1e-6 # used to balance the style loss and content loss
'''config file'''


'''dataset'''
datasetpath = '' # use coco(http://cocodataset.org/#home) in original paper
style_image_path = 'dataset/styles/candy.jpg'


'''train'''
num_epochs = 10
batch_size = 4
image_size = (256, 256)
learning_rate = 1e-3
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225] # used for pytorch pretrained models
backupdir = 'backup'
save_interval = 500 # every save_interval batchs, save the model
logfilepath = 'train.log'
checkpointpath_restore = ''
 
__all__ = ['r3d_18', 'se_r3d_18','da_18','da_se_18','mc3_18', 'r2plus1d_18','r3d_34','vgg16_3d','vgg19_3d',\
          'se_mc3_18','da_mc3_18','da_se_mc3_18']

from utils import get_weight_path

NET_NAME = 'da_se_18'
VERSION = 'v4'
DEVICE = '3'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 0,1,2,3,4
CURRENT_FOLD = 0
GPU_NUM = len(DEVICE.split(','))


# WEIGHT_PATH = {
#   'r3d_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:28-train_loss:0.15993-val_loss:0.31931.pth'.format(VERSION),#1.0
#   'se_r3d_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:24-train_loss:0.14345-val_loss:0.14227.pth'.format(VERSION),#4.3
#   'da_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:27-train_loss:0.14407-val_loss:0.14218.pth'.format(VERSION),#10.1
#   'da_se_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:32-train_loss:0.13142-val_loss:0.15287.pth'.format(VERSION),#11.0
# }
'''
WEIGHT_PATH_DICT = {
  'r3d_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:7-train_loss:0.45322-val_loss:0.35240.pth'.format(VERSION),
  'se_r3d_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:15-train_loss:0.27431-val_loss:0.19606.pth'.format(VERSION),
  'da_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:7-train_loss:0.32471-val_loss:0.25842.pth'.format(VERSION),
  'da_se_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:32-train_loss:0.13142-val_loss:0.15287.pth'.format(VERSION),
}

WEIGHT_PATH = WEIGHT_PATH_DICT[NET_NAME]
'''

# CKPT_PATH = './new_ckpt/{}'.format(VERSION)
CKPT_PATH = './new_ckpt/{}'.format(VERSION)
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)


# Arguments when trainer initial
INIT_TRAINER = {
  'net_name':NET_NAME,
  'lr':1e-3, 
  'n_epoch':50,
  'channels':1,
  'num_classes':3,
  'input_shape':(64,224,224),
  'crop':None,
  'batch_size':12,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'weight_path':WEIGHT_PATH,
  'weight_decay': 0.,
  'momentum': 0.9,
  'gamma': 0.1,
  'milestones': [40,80],
  'T_max':5,
  'use_fp16':True
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
  'output_dir':'./new_ckpt/{}'.format(VERSION),
  'log_dir':'./new_log/{}'.format(VERSION),
  'optimizer':'Adam',
  'loss_fun':'Cross_Entropy',
  'class_weight':None,
  'lr_scheduler':'CosineAnnealingLR'
  }


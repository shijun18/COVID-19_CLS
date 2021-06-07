 
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


'''
WEIGHT_PATH_DICT = {
  'r3d_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:7-train_loss:0.45322-val_loss:0.35240.pth'.format('v1.4'),
  'se_r3d_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:15-train_loss:0.27431-val_loss:0.19606.pth'.format('v4.4'),
  'da_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:7-train_loss:0.32471-val_loss:0.25842.pth'.format('v10.4'),
  'da_se_18':'/staff/shijun/torch_projects/COVID-19_CLS/ckpt/{}/epoch:32-train_loss:0.13142-val_loss:0.15287.pth'.format('v11.0'),
}

WEIGHT_PATH = WEIGHT_PATH_DICT[NET_NAME]
print(WEIGHT_PATH)
'''
# CKPT_PATH = './new_ckpt/{}'.format(VERSION)
CKPT_PATH = './tmp_ckpt/{}'.format(VERSION)
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)


# Arguments when trainer initial
INIT_TRAINER = {
  'net_name':NET_NAME,
  'lr':1e-3, 
  'n_epoch':100,
  'channels':1,
  'num_classes':3,
  'input_shape':(64,224,224),
  'crop':None,
  'batch_size':6,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'weight_path':WEIGHT_PATH,
  'weight_decay': 0.,
  'momentum': 0.9,
  'gamma': 0.1,
  'milestones': [30,60],
  'T_max':5,
  'use_fp16':False
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
  'output_dir':'./tmp_ckpt/{}'.format(VERSION),
  'log_dir':'./tmp_log/{}'.format(VERSION),
  'optimizer':'Adam',
  'loss_fun':'Cross_Entropy',
  'class_weight':None,
  'lr_scheduler':'MultiStepLR'
  }


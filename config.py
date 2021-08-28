 
__all__ = ['r3d_18', 'se_r3d_18','da_18','da_se_18','r3d_34','se_r3d_34','da_34','da_se_34','vgg16_3d','vgg19_3d']

from utils import get_weight_path,get_weight_list

NET_NAME = 'r3d_34'
VERSION = 'v5.0'
DEVICE = '0,1'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4,5
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5


CKPT_PATH = './tmp_ckpt/{}/fold{}'.format(VERSION,CURRENT_FOLD)
WEIGHT_PATH = get_weight_path(CKPT_PATH)
# print(WEIGHT_PATH)

if PRE_TRAINED:
    WEIGHT_PATH_LIST = get_weight_list('./tmp_ckpt/{}/'.format(VERSION))
else:
    WEIGHT_PATH_LIST = None

# Arguments when trainer initial
INIT_TRAINER = {
  'net_name':NET_NAME,
  'lr':1e-3, 
  'n_epoch':80,
  'channels':1,
  'num_classes':3,
  'input_shape':(64,224,224),
  'crop':None,
  'batch_size':8,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'weight_path':WEIGHT_PATH,
  'weight_decay': 0.,
  'momentum': 0.9,
  'gamma': 0.1,
  'milestones': [30,60,90],
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
  'lr_scheduler':'MultiStepLR' # MultiStepLR
  }


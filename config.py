 
__all__ = ['r3d_18', 'mc3_18', 'r2plus1d_18','se_r3d_18','vgg16_3d','vgg19_3d',\
          'se_mc3_18','r3d_conv_18']


NET_NAME = 'r3d_conv_18'
VERSION = 'v8.0'
DEVICE = '2'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4,5
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))


WEIGHT_PATH = {
  'r3d_18':'./ckpt/{}/epoch:18-train_loss:0.21740-val_loss:0.16584.pth'.format(VERSION),
  'mc3_18':'./ckpt/{}/epoch:17-train_loss:0.31278-val_loss:0.23510.pth'.format(VERSION),
  'r2plus1d_18':'./ckpt/{}/epoch:14-train_loss:0.31134-val_loss:1.51052.pth'.format(VERSION),
  'se_r3d_18':'./ckpt/{}/epoch:24-train_loss:0.21298-val_loss:0.16938.pth'.format(VERSION),
  'vgg16_3d':'./ckpt/{}/epoch:37-train_loss:0.13268-val_loss:0.13816.pth'.format(VERSION),
  'vgg19_3d':'./ckpt/{}/epoch:34-train_loss:0.14165-val_loss:0.11818.pth'.format(VERSION),
  'se_mc3_18':'./ckpt/{}/epoch:38-train_loss:0.24484-val_loss:0.21495.pth'.format(VERSION),
  'r3d_conv_18':'./ckpt/{}/'.format(VERSION)
}

# Arguments when trainer initial
INIT_TRAINER = {
  'net_name':NET_NAME,
  'lr':1e-3, 
  'n_epoch':100,
  'channels':1,
  'num_classes':3,
  'input_shape':(64,224,224),
  'crop':48,
  'batch_size':6,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'weight_path':WEIGHT_PATH[NET_NAME],
  'weight_decay': 0.,
  'momentum': 0.9,
  'gamma': 0.1,
  'milestones': [40,80],
  'T_max':5,
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
  'output_dir':'./ckpt/{}'.format(VERSION),
  'log_dir':'./log/{}'.format(VERSION),
  'optimizer':'Adam',
  'loss_fun':'Cross_Entropy',
  'class_weight':None,
  'lr_scheduler':None
  }


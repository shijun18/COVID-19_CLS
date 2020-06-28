 
__all__ = ['r3d_18', 'mc3_18', 'r2plus1d_18','se_r3d_18','vgg16_3d','vgg19_3d','se_mc3_18']


NET_NAME = 'se_mc3_18'
VERSION = 'v7.2'
DEVICE = '6,7'
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
  'vgg16_3d':'./ckpt/{}/epoch:19-train_loss:0.24623-val_loss:0.23104.pth'.format(VERSION),
  'vgg19_3d':'./ckpt/{}/epoch:18-train_loss:0.26500-val_loss:0.19258.pth'.format(VERSION),
  'se_mc3_18':'./ckpt/{}/'.format(VERSION)
}

# Arguments when trainer initial
INIT_TRAINER = {
  'net_name':NET_NAME,
  'lr':1e-3, 
  'n_epoch':40,
  'channels':1,
  'num_classes':3,
  'input_shape':(64,224,224),
  'crop':48,
  'batch_size':5,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'weight_path':WEIGHT_PATH[NET_NAME]
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
  'output_dir':'./ckpt/{}'.format(VERSION),
  'log_dir':'./log/{}'.format(VERSION),
  'optimizer':'SGD',
  'loss_fun':'Cross_Entropy',
  'class_weight':None,
  'lr_scheduler':None
  }


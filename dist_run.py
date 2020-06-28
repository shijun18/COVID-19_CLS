import os
import sys
import argparse
from dist_trainer import Dist_VolumeClassifier
import torch.multiprocessing as mp
from data_utils.csv_reader import csv_reader_single
from config import INIT_TRAINER,SETUP_TRAINER,VERSION

import time

def get_cross_validation(path_list,fold_num,current_fold):
  
  _len_ = len(path_list) // fold_num
  train_id = []
  validation_id = []
  end_index = current_fold * _len_
  start_index = end_index - _len_
  if current_fold == fold_num:
    validation_id.extend(path_list[start_index:])
    train_id.extend(path_list[:start_index])
  else:
    validation_id.extend(path_list[start_index:end_index])
    train_id.extend(path_list[:start_index])
    train_id.extend(path_list[end_index:])
  
  print(len(train_id),len(validation_id))
  return train_id,validation_id 


def get_parameter_number(net):
  total_num = sum(p.numel() for p in net.parameters())
  trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
  return {'Total': total_num, 'Trainable': trainable_num}






if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--nodes', default=1,
                      type=int, metavar='N')
  parser.add_argument('-g', '--gpus', default=1, type=int,
                      help='number of gpus per node')
  parser.add_argument('-nr', '--nr', default=0, type=int,
                      help='ranking within the nodes')

  args = parser.parse_args()
  world_size = args.gpus * args.nodes
  os.environ['MASTER_ADDR'] = '222.195.72.216'              
  os.environ['MASTER_PORT'] = '11111' 

  csv_path = './converter/shuffle_label.csv'
  label_dict = csv_reader_single(csv_path,key_col='id',value_col='label')
  path_list = list(label_dict.keys())[:3600] 
  train_path,val_path = get_cross_validation(path_list,5,1)
  trainer = Dist_VolumeClassifier(**INIT_TRAINER)
  
  print(get_parameter_number(trainer.net))
  start_time = time.time()
  
  add_argument = (args.gpus,args.nr,world_size,train_path,val_path,label_dict)
  mp.spawn(trainer.trainer,args=add_argument + tuple(SETUP_TRAINER.values()),nprocs=args.gpus)
  
  print('%.4f'%(time.time()-start_time))
  
  
  
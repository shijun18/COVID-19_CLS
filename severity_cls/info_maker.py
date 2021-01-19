import sys
sys.path.append('..')
import os
import pandas as pd 
import numpy as np
from trainer import VolumeClassifier
from data_utils.csv_reader import csv_reader_single
from config import INIT_TRAINER,VERSION


def clean_data(csv_path,data_path,save_path):
  csv_file = pd.read_csv(csv_path)
  id_list = csv_file['patient_id'].values.tolist()
  id_list = [str(case) for case in id_list]
  print(len(id_list))
  path_list = os.listdir(data_path)
  target_list = [os.path.basename(os.path.splitext(case)[0]) for case in path_list]
 
  except_list = []
  except_index = []

  for index,item in enumerate(id_list):
    if item not in target_list:
      except_list.append(item)
      except_index.append(index)
      # print(item)
    
  new_csv = csv_file.drop(except_index)
  print(len(new_csv['patient_id'].values.tolist()))
  new_csv.to_csv(save_path,index=False)
  print(len(except_list)) 
  


if __name__ == "__main__":
  
  # Clean the data without CT images 
  # csv_path = './modified_clincal_info.csv'
  # data_path = '../dataset/npy_data/NCP'
  # save_path = './clean_clinal_info.csv'
  # clean_data(csv_path,data_path,save_path)
  ###############################################
  
  # Add the CT score produced by CNN for each sample
  csv_path = './input_file/clean_clinal_info.csv'
  data_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/npy_data/NCP'
  save_path = './input_file/{}_score_clincal_info.csv'.format(VERSION)
  csv_file = pd.read_csv(csv_path)
  
  id_list = csv_file['patient_id'].values.tolist()
  path_list = [os.path.join(data_path,str(case)+'.hdf5') for case in id_list]
  print(len(path_list))

  label_path = '../converter/shuffle_label.csv'
  label_dict = csv_reader_single(label_path,key_col='id',value_col='label')

  classifier = VolumeClassifier(**INIT_TRAINER)
  result,feature_in,feature_out = classifier.inference(path_list,label_dict,hook_fn_forward=True)
  csv_file['Score'] = [round(case[1],4) for case in result['prob']]
  print(len(csv_file['Score']))
  csv_file['true'] = result['true']
  csv_file['prob'] = result['prob']

  csv_file.to_csv(save_path,index=False)
  ###############################################
  # Save feature
  # feature_dir = './mid_feature/{}'.format(VERSION)
  # if not os.path.exists(feature_dir):
  #   os.makedirs(feature_dir)
  # from converter.common_utils import save_as_hdf5
  # for i in range(len(id_list)):
  #   name = str(id_list[i])+'.hdf5'
  #   feature_path = os.path.join(feature_dir,name)
  #   save_as_hdf5(feature_in[i],feature_path,'feature_in')   
  #   save_as_hdf5(feature_out[i],feature_path,'feature_out') 
  ###############################################  
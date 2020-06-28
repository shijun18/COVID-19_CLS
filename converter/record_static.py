import os
import glob
import pandas as pd
import random

from common_utils import hdf5_reader

RULE = {"CP":0,
        "NCP":1,
        "Normal":2
        }

def make_label_csv(input_path,csv_path):
  '''
  Make label csv file.
  label rule: CP->0, NCP->1, Normal->2
  '''
  info = []
  for subdir in os.scandir(input_path):
    index = RULE[subdir.name]
    path_list = glob.glob(os.path.join(subdir.path,"*.hdf5"))
    sub_info = [[item,index] for item in path_list]
    info.extend(sub_info)
  
  col = ['id','label']
  info_data = pd.DataFrame(columns=col,data=info)
  info_data.to_csv(csv_path,index=False)



def random_shuffle_csv(source_file,save_file):
  '''
  Shuffle the rows of the csv randomly, return a new csv.
  '''
  file_csv = pd.read_csv(source_file)
  new_csv = []
  col = list(file_csv.columns.values)
  for _,line in file_csv.iterrows():
    new_csv.append([line[name] for name in col])
  
  random.shuffle(new_csv)
  
  new_file = pd.DataFrame(columns=col,data=new_csv)
  new_file.to_csv(save_file,index=False)




def statistic_slice_num(input_path,csv_path):
  '''
  Count the slice number for per sample.
  '''
  info = []
  for subdir in os.scandir(input_path):
    path_list = glob.glob(os.path.join(subdir.path,"*.hdf5"))
    sub_info = [[item,hdf5_reader(item,'img').shape[0]] for item in path_list]
    info.extend(sub_info)
  
  col = ['id','slice_num']
  info_data = pd.DataFrame(columns=col,data=info)
  info_data.to_csv(csv_path,index=False)



if __name__ == "__main__":
  
  '''
  # Part-1: make label csv file
  input_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/npy_data/'
  csv_path = './label.csv'
  make_label_csv(input_path,csv_path)
  '''
  
  # Part-2: random shuffle the label csv file
  input_path = './label.csv'
  save_path = './shuffle_label.csv'
  random_shuffle_csv(input_path,save_path)


  '''
  # Part-3: Count the slice number
  input_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/npy_data/'
  csv_path = './slice_number.csv'
  statistic_slice_num(input_path,csv_path)
  '''
import os
import glob
import pandas as pd
import random
from tqdm import tqdm

from common_utils import hdf5_reader

RULE = {"CP": 0, "NCP": 1, "Normal": 2}


def make_label_csv(input_path, csv_path):
    '''
    Make label csv file.
    label rule: CP->0, NCP->1, Normal->2
    '''
    info = []
    for subdir in os.scandir(input_path):
        index = RULE[subdir.name]
        path_list = glob.glob(os.path.join(subdir.path, "*.hdf5"))
        sub_info = [[item, index] for item in path_list]
        info.extend(sub_info)

    col = ['id', 'label']
    random.shuffle(info)
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(csv_path, index=False)


def statistic_slice_num(input_path, csv_path):
    '''
    Count the slice number for per sample.
    '''
    info = []
    for subdir in os.scandir(input_path):
        path_list = glob.glob(os.path.join(subdir.path, "*.hdf5"))
        sub_info = [[item, hdf5_reader(item, 'img').shape[0]]
                    for item in path_list]
        info.extend(sub_info)

    col = ['id', 'slice_num']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(csv_path, index=False)


def count_set(input_path):

    slice_num = {
        'CP': [], 
        'NCP': [], 
        'Normal':[]
        }

    for item in tqdm(input_path):
        if 'Normal' in item:
            slice_num['Normal'].append(hdf5_reader(item, 'img').shape[0])
        elif 'NCP' in item:
            slice_num['NCP'].append(hdf5_reader(item, 'img').shape[0])
        else:
            slice_num['CP'].append(hdf5_reader(item, 'img').shape[0])

    return slice_num


if __name__ == "__main__":
  
  # Part-1: make label csv file
  input_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/resize_data/'
  csv_path = './new_resize_shuffle_label.csv'
  make_label_csv(input_path,csv_path)
  
  '''
  # Part-2: Count the slice number
  input_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/npy_data/'
  csv_path = './new_slice_number.csv'
  statistic_slice_num(input_path,csv_path)
  '''
  '''
  csv_path = './shuffle_label.csv'
  print(csv_path)
  path_list = pd.read_csv(csv_path)['id'].values.tolist()[:3600]
  slice_num = count_set(path_list)
  print(len(slice_num['Normal']), sum(slice_num['Normal']))
  print(len(slice_num['NCP']), sum(slice_num['NCP']))
  print(len(slice_num['CP']), sum(slice_num['CP']))

  path_list = pd.read_csv(csv_path)['id'].values.tolist()[3600:]
  slice_num = count_set(path_list)
  print(len(slice_num['Normal']), sum(slice_num['Normal']))
  print(len(slice_num['NCP']), sum(slice_num['NCP']))
  print(len(slice_num['CP']), sum(slice_num['CP']))
  '''
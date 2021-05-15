import os 
import pandas as pd
import glob
import numpy as np
from skimage.transform import resize
from common_utils import hdf5_reader,save_as_hdf5
import random


DIM = (64,224,224)
CROP = 48
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
    random.shuffle(info)
    info_data = pd.DataFrame(columns=col,data=info)
    info_data.to_csv(csv_path,index=False)



def resize_data(input_path,save_path):
    '''
    resize the data to fixed size.
    '''
    if os.path.isdir(input_path):
        entry = os.scandir(input_path)
        for item in entry:
            if item.is_dir():
                temp_path = os.path.join(save_path,item.name)
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                resize_data(item.path,temp_path)

            elif item.is_file() and os.path.splitext(item.name)[1] == '.hdf5':
                img = hdf5_reader(item.path,'img')
                img = img[:,CROP:-CROP,CROP:-CROP]
                resize_img = resize(img,DIM,anti_aliasing=True)
                data_path = os.path.join(save_path,item.name)
                save_as_hdf5(resize_img,data_path,'img')


if __name__ == "__main__":
    input_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/npy_data'
    save_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/resize_data'
    resize_data(input_path,save_path)
    csv_path = './new_resize_shuffle_label.csv'
    make_label_csv(save_path,csv_path)
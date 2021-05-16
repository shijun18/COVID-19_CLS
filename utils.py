import shutil
import os
import pandas as pd

def make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def remove_dir(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)

def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None


def exclude_path(source_csv,dest_csv,key='id'):
    source_item = pd.read_csv(source_csv)[key].values.tolist()
    print('source len:',len(source_item))
    dest_item = pd.read_csv(dest_csv)[key].values.tolist()
    print('dest len:',len(dest_item))
    exclude_item = []
    for item in dest_item:
        if item in source_item:
            continue
        else:
            exclude_item.append(item)
    
    print('exclude len:',len(exclude_item))

    return exclude_item


if __name__ == '__main__':
    ex_path = exclude_path('./converter/shuffle_label.csv', './converter/new_shuffle_label.csv','id')
    
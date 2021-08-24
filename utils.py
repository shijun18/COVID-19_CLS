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


def get_weight_list(ckpt_path,choice=None):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if choice is not None and eval(str(fold.name)[-1]) not in choice:
            continue
        if fold.is_dir():
            weight_path = os.listdir(fold.path)
            # print(weight_path)
            weight_path.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            path_list.append(os.path.join(fold.path,weight_path[-1]))
            # path_list.append(os.path.join(fold.path,weight_path[0]))
            # print(os.path.join(fold.path,weight_path[-1]))
    path_list.sort(key=lambda x:x.split('/')[-2])
    return path_list


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


def remove_weight_path(ckpt_path,retain=5):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=5):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain=retain)
        else:
            remove_weight_path(ckpt_path,retain=retain)
            break  


if __name__ == '__main__':
    # ex_path = exclude_path('./converter/shuffle_label.csv', './converter/new_shuffle_label.csv','id')
    ckpt_path = './new_ckpt/'
    dfs_remove_weight(ckpt_path)
    
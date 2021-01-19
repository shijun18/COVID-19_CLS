import shutil
import os

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
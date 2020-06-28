import os
import glob
import h5py
import numpy as np
import zipfile

def unzip_single(input_path,save_path):
  '''
  Unzip single .zip file to destination directory.
  '''
  try:
    zf = zipfile.ZipFile(input_path)
    zf.extractall(path=save_path)
    zf.close() 
  except:
    print(input_path)
    pass 


def unzip_data(input_path,save_path):
  '''
  Unzip the raw data (.zip) and generate the same directory structure with original directory.
  '''
  if os.path.isdir(input_path):
    entry_iterator = os.scandir(input_path)
    for item in entry_iterator:
      if item.is_dir():
        temp_path = os.path.join(save_path,item.name)
        if not os.path.exists(temp_path):
          os.makedirs(temp_path)
        unzip_data(item.path,temp_path)  

      elif item.is_file() and os.path.splitext(item.name)[1] == '.zip':
        name = os.path.splitext(item.name)[0]
        temp_path = os.path.join(save_path,name)
        if not os.path.exists(temp_path):
          os.makedirs(temp_path)
        unzip_single(item.path,temp_path)
        print("%s done!" % item.path)
  
  elif os.path.isfile(input_path) and os.path.splitext(os.path.basename(input_path))[1] == '.zip':
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    unzip_single(input_path,save_path)
    print("%s done!" % input_path)



if __name__ == "__main__":
    
  input_path = "/staff/shijun/dataset/Covid-19_CT"
  save_path = "/staff/shijun/torch_projects/COVID-19_CLS/dataset/raw_data"
  
  # input_path = "/staff/shijun/dataset/Covid-19_CT/NCP/COVID19-2.zip"
  # save_path = "/staff/shijun/torch_projects/COVID-19_CLS/dataset/raw_data/NCP/COVID19-2"

  # input_path = "/staff/shijun/dataset/Covid-19_CT/CP/CP-1.zip"
  # save_path = "/staff/shijun/torch_projects/COVID-19_CLS/dataset/raw_data/CP/CP-1"
  
  unzip_data(input_path,save_path)
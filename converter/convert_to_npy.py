import os
import numpy as np
from PIL import Image

from common_utils import save_as_hdf5


POSTFIX = ['.png','.jpg','.JPG','.Jpg','.jpeg','.bmp','.BMP','.tif']
DIM = (512,512)


def postfix_search(input_path):
  '''
    DFS for postfix searching which is beneficial for data converting.
  '''
  postfix = set()
  if os.path.isdir(input_path):
    entry_iterator = os.scandir(input_path)
    for item in entry_iterator:
      if item.is_dir():
        postfix = postfix.union(postfix_search(item.path))

      else:
        postfix.add(os.path.splitext(item.name)[1])

    return postfix    



def convert_to_npy(input_path,save_path):
  '''
  Convert the raw data(e.g. jpg...) to numpy array and save as hdf5.
  
  Basic process operations:
  - normalization:[0,1]
  - resize:(512,512)
  - stack:single silce to 3d format
  '''
  ID = []
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  
  if os.path.isdir(input_path):
    item_list = os.listdir(input_path)
   
    if len(item_list) > 0:
      if os.path.isfile(os.path.join(input_path,item_list[0])): 
        patient_id = os.path.basename(input_path)
        ID.append(patient_id)
        hdf5_path = os.path.join(save_path,"%s.hdf5" % patient_id)
        
        try:
          # convert image to numpy array with fixed 2d-dim: DIM(512,512)
          item_list.sort(key=lambda x:int(x.split('.')[0]))
          img_list = [img_reader(os.path.join(input_path,item),DIM) for item in item_list]
          img_array = np.stack(img_list,axis=0) # (z,x,y)

          # save as hdf5, key='img'
          save_as_hdf5(img_array,hdf5_path,'img')

        except:
          print(input_path)
          pass

      else:
        for item in item_list:
          ID.extend(convert_to_npy(os.path.join(input_path,item),save_path))

  return ID 



def img_reader(input_path,dim):
  '''
  Image file reader, return image array.

  Other operation:
  - resize: fixed dim
  - normalize: [0,1]

  Args:
  - input path: file path
  - dim: a tuple of 2 integers
  '''
  # graylevel mode
  img = Image.open(input_path).convert('L')

  # resize if need, mode=Image.NEAREST
  if img.size != dim:
      img = img.resize(dim,Image.NEAREST)

  # convert to numpy array, data type = np.float32    
  img_array = np.asarray(img,dtype=np.float32)
  
  # normalize:[0,255] -> [0.0,1.0]
  img_array = img_array / 255.0

  return img_array




if __name__ == "__main__":
  
  '''
  # Part-1:search all file postfixes for converting 

  input_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/raw_data/Normal'
  postfix = postfix_search(input_path)
  print(postfix)
  '''

  
  # Part-2:convert image to numpy array and save as hdf5
  input_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/raw_data/CP'
  save_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/npy_data/CP'
  patient_id = convert_to_npy(input_path,save_path)
  print("CP %d samples done"%len(patient_id))

  input_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/raw_data/NCP'
  save_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/npy_data/NCP'
  patient_id = convert_to_npy(input_path,save_path)
  print("NCP %d samples done"%len(patient_id))


  input_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/raw_data/Normal'
  save_path = '/staff/shijun/torch_projects/COVID-19_CLS/dataset/npy_data/Normal'
  patient_id = convert_to_npy(input_path,save_path)
  print("Normal %d samples done"%len(patient_id))
 
  
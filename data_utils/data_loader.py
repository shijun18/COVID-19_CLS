import sys
sys.path.append('..')
from converter.common_utils import hdf5_reader
from torch.utils.data import Dataset
import torch


class DataGenerator(Dataset):
  '''
  Custom Dataset class for data loader.
  Args：
  - path_list: list of file path
  - label_dict: dict, file path as key, label as value
  - transform: the data augmentation methods
  '''
  def __init__(self, path_list, label_dict, transform=None):

    self.path_list = path_list
    self.label_dict = label_dict
    self.transform = transform


  def __len__(self):
    return len(self.path_list)


  def __getitem__(self,index):
    # Get image and label
    # image: D,H,W
    # label: integer, 0,1,..
    image = hdf5_reader(self.path_list[index],'img')
    # print(self.path_list[index])
    # assert len(image.shape) == 3
    label = self.label_dict[self.path_list[index]]    
    sample = {'image':image, 'label':int(label)}
    # Transform
    if self.transform is not None:
      sample = self.transform(sample)
    
    return sample
      


class data_prefetcher():
  def __init__(self, loader):
    self.loader = iter(loader)
    self.stream = torch.cuda.Stream()
    self.preload()

  def preload(self):
    try:
      self.next_input, self.next_target = next(self.loader)
    except StopIteration:
      self.next_input = None
      self.next_target = None
      return
    with torch.cuda.stream(self.stream):
      self.next_input = self.next_input.cuda(non_blocking=True)
      self.next_target = self.next_target.cuda(non_blocking=True)
        
  def next(self):
    torch.cuda.current_stream().wait_stream(self.stream)
    input = self.next_input
    target = self.next_target
    self.preload()
    return input, target     
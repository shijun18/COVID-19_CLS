import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import pandas as pd
import numpy as np
import math

import data_utils.transform as tr
from data_utils.data_loader import DataGenerator


# Distributed GPU version.

class Dist_VolumeClassifier(object):
  '''
  Control the training, evaluation, and inference process.
  Args:
  - net_name: string, __all__ = ['r3d_18', 'mc3_18', 'r2plus1d_18'].
  - lr: float, learning rate.
  - n_epoch: integer, the epoch number
  - channels: integer, the channel number of the input
  - num_classes: integer, the number of class
  - input_shape: tuple of integer, input dim
  - crop: integer, cropping size
  - batch_size: integer
  - num_workers: integer, how many subprocesses to use for data loading.
  - device: string, use the specified device
  - pre_trained: True or False, default False
  - weight_path: weight path of pre-trained model
  '''
  def __init__(self,net_name=None,lr=1e-3,n_epoch=1,channels=1,num_classes=3,input_shape=None,crop=48,
                batch_size=6,num_workers=0,device=None,pre_trained=False,weight_path=None): 
    super(Dist_VolumeClassifier,self).__init__()    

    self.net_name = net_name
    self.lr = lr
    self.n_epoch = n_epoch
    self.channels = channels
    self.num_classes = num_classes
    self.input_shape = input_shape
    self.crop = crop
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.device = device
    self.net = self._get_net(self.net_name)
    self.start_epoch = 0
    self.global_step = 0
    self.loss_threshold = 1.0
    self.pre_trained = pre_trained
    self.weight_path = weight_path
    
    os.environ['CUDA_VISIBLE_DEVICES'] = self.device
    
    if self.pre_trained:
      self._get_pre_trained(self.weight_path)
      self.loss_threshold = eval(os.path.splitext(self.weight_path.split(':')[-1])[0])
  
  def trainer(self,gpu_id,gpu_num,node_id,world_size,train_path,val_path,label_dict,output_dir=None,log_dir=None,optimizer='Adam',
                loss_fun='Cross_Entropy',class_weight=None,lr_scheduler=None):
    '''
    Note: gpu_id for distributed training
    Args:
    - gpu_id: integer, gpu index within current node
    - gpu_num: integer, gpu number per node
    - node_id: interger, current node index
    - world_size: interger, total number of process, one gpu to one process
    '''
    self.gpu_id = gpu_id
    self.rank = node_id*gpu_num + gpu_id
    self.world_size = world_size
    self.gpu_num = gpu_num
    torch.cuda.set_device(self.gpu_id)
    torch.manual_seed(0)
    self.global_step = self.start_epoch * math.ceil(len(train_path)/self.gpu_num/self.batch_size)

    print('GPU:{}'.format(self.gpu_id))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(log_dir):
      os.makedirs(log_dir)

    self.writer = SummaryWriter(log_dir)
    
    net = self.net
    lr = self.lr
    loss = self._get_loss(loss_fun,class_weight) 
    
    # copy to gpu
    net = net.cuda(self.gpu_id)
    loss = loss.cuda(self.gpu_id)

    # dataloader setting
    train_transformer = transforms.Compose([
      tr.CropResize(dim=self.input_shape,crop=self.crop),
      tr.RandomTranslationRotationZoom(mode='trz'),
      tr.RandomFlip(mode='hv'),
      tr.To_Tensor(n_class=self.num_classes)
    ])

    train_dataset = DataGenerator(train_path,label_dict,transform=train_transformer)
    shuffle = True
    train_sampler = None
    if self.gpu_num > 1:
      train_sampler = DistributedSampler(train_dataset,
                                        num_replicas=self.world_size,
                                        rank=self.rank)
      dist.init_process_group(backend='nccl',
                              init_method='env://',
                              world_size=self.world_size,
                              rank=self.rank)
      net = DistributedDataParallel(net,device_ids=[self.gpu_id])
      shuffle=False
      
    
    train_loader = DataLoader(
      train_dataset,
      batch_size=self.batch_size,
      shuffle=shuffle,
      num_workers=self.num_workers,
      sampler=train_sampler,
      pin_memory=True
    )


    # optimizer setting
    optimizer = self._get_optimizer(optimizer,net,lr)
    if self.pre_trained:
      checkpoint = torch.load(self.weight_path)
      optimizer.load_state_dict(checkpoint['optimizer']) 

    if lr_scheduler is not None:
      lr_scheduler = self._get_lr_scheduler(lr_scheduler,optimizer)


    
    for epoch in range(self.start_epoch,self.n_epoch):
      train_loss,train_acc = self._train_on_epoch(epoch,net,loss,optimizer,train_loader)
      
      torch.cuda.empty_cache()

      if lr_scheduler is not None:
        lr_scheduler.step(val_loss)

      if self.rank == 0:

        val_loss,val_acc = self._val_on_epoch(epoch,net,loss,val_path,label_dict)
        
        print('epoch:{},train_loss:{:.5f},val_loss:{:.5f}'
          .format(epoch,train_loss,val_loss))
      
        print('epoch:{},train_acc:{:.5f},val_acc:{:.5f}'
          .format(epoch,train_acc,val_acc))

        self.writer.add_scalars(
          'data/loss',{'train':train_loss,'val':val_loss},epoch
        )
        self.writer.add_scalars(
          'data/acc',{'train':train_acc,'val':val_acc},epoch
        )
        self.writer.add_scalar(
          'data/lr',optimizer.param_groups[0]['lr'],epoch
        )

        if not os.path.exists(output_dir):
          os.makedirs(output_dir)
        
        if val_loss < self.loss_threshold:
          self.loss_threshold = val_loss
          
          if len(self.device.split(',')) > 1:
            state_dict = net.module.state_dict()
          else:
            state_dict = net.state_dict()

          saver = {
            'epoch':epoch,
            'save_dir':output_dir,
            'state_dict':state_dict,
            'optimizer':optimizer.state_dict()
          }  
          
          file_name = 'epoch:{}-train_loss:{:.5f}-val_loss:{:.5f}.pth'.format(epoch,train_loss,val_loss) 
          save_path = os.path.join(output_dir,file_name)

          torch.save(saver,save_path)

        
    self.writer.close()


  def _train_on_epoch(self,epoch,net,criterion,optimizer,train_loader):

    net.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    for step,sample in enumerate(train_loader):
      data = sample['image']
      target = sample['label']

      data = data.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)

      output = net(data)
      loss = criterion(output,target)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


      output = output.float()
      loss = loss.float()
      
      # measure accuracy and record loss
      acc = accuracy(output.data,target)[0]
      train_loss.update(loss.item(),data.size(0))
      train_acc.update(acc.item(),data.size(0))

      torch.cuda.empty_cache()
      
      print('rank:{},epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'
        .format(self.rank,epoch,step,loss.item(),acc.item(),optimizer.param_groups[0]['lr']))
      if self.rank == 0 and self.global_step%10==0:
        self.writer.add_scalars(
          'data/train_loss_acc',{'train_loss':loss.item(),'train_acc':acc.item()},self.global_step
        )

      self.global_step += 1  

    return train_loss.avg,train_acc.avg
  


  def _val_on_epoch(self,epoch,net,criterion,val_path,label_dict):

    net.eval()
    
    val_transformer = transforms.Compose([
      tr.CropResize(dim=self.input_shape,crop=self.crop),
      tr.To_Tensor(n_class=self.num_classes)
    ])

    val_dataset = DataGenerator(val_path,label_dict,transform=val_transformer)

    val_loader = DataLoader(
      val_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=True
    )

    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    with torch.no_grad():
      for step,sample in enumerate(val_loader):
        data = sample['image']
        target = sample['label']

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = net(data)
        loss = criterion(output,target)

        output = output.float()
        loss = loss.float()
        
        # measure accuracy and record loss
        acc = accuracy(output.data,target)[0]
        val_loss.update(loss.item(),data.size(0))
        val_acc.update(acc.item(),data.size(0))

        torch.cuda.empty_cache()

        print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f}'
          .format(epoch,step,loss.item(),acc.item()))
      

    return val_loss.avg,val_acc.avg

  


  def inference(self,test_path,label_dict,save_path,net=None):
    
    if net is None:
      net = self.net

    net.eval()

    test_transformer = transforms.Compose([
      tr.CropResize(dim=self.input_shape,crop=self.crop),
      tr.To_Tensor(n_class=self.num_classes)
    ])

    test_dataset = DataGenerator(test_path,label_dict,transform=test_transformer)

    test_loader = DataLoader(
      test_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=True
    )
    
    result = {
      'true':[],
      'pred':[],
      'prob':[]
    }

    test_acc = AverageMeter()

    with torch.no_grad():
      for step,sample in enumerate(test_loader):
        data = sample['image']
        target = sample['label']

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True) #N

        output = net(data)
        output = output.float() #N*C
        
        acc = accuracy(output.data,target)[0]
        test_acc.update(acc.item(),data.size(0))

        result['true'].extend(target.detach().tolist())
        result['pred'].extend(torch.argmax(output,1).detach().tolist())
        result['prob'].extend(output.detach().tolist())
        
        print('step:{},test_acc:{:.5f}'
          .format(step,acc.item()))

        torch.cuda.empty_cache()
    
    print('average test_acc:{:.5f}'.format(test_acc.avg))

    csv_file = pd.DataFrame(result)
    csv_file.to_csv(save_path,index=False)  
    
    


  def _get_net(self,net_name):
    if net_name == 'r3d_18':
      from model.resnet_3d import r3d_18
      net = r3d_18(input_channels=self.channels,num_classes=self.num_classes)

    elif net_name == 'mc3_18':
      from model.resnet_3d import mc3_18
      net = mc3_18(input_channels=self.channels,num_classes=self.num_classes)

    elif net_name == 'r2plus1d_18':
      from model.resnet_3d import r2plus1d_18
      net = r2plus1d_18(input_channels=self.channels,num_classes=self.num_classes)   
    
    elif net_name == 'se_mc3_18':
      from model.se_resnet_3d import se_mc3_18
      net = se_mc3_18(input_channels=self.channels,num_classes=self.num_classes)

    return net  


  def _get_loss(self,loss_fun,class_weight=None):
    if class_weight is not None:
      class_weight = torch.tensor(class_weight)

    if loss_fun == 'Cross_Entropy':  
      loss = nn.CrossEntropyLoss(class_weight)  

    return loss  


  def _get_optimizer(self,optimizer,net,lr):
    if optimizer == 'Adam':
      optimizer = torch.optim.Adam(net.parameters(),lr=lr)  

    elif optimizer == 'SGD':
      optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9)

    return optimizer   


  def _get_lr_scheduler(self,lr_scheduler,optimizer):
    if lr_scheduler == 'ReduceLROnPlateau':
      lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                       mode='min',patience=5,verbose=True)
    
    return lr_scheduler
  
  
  def _get_pre_trained(self,weight_path):
    checkpoint = torch.load(weight_path)
    self.net.load_state_dict(checkpoint['state_dict']) 
    self.start_epoch = checkpoint['epoch'] + 1
   



# computing tools

class AverageMeter(object):
  '''
  Computes and stores the average and current value
  '''
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
  '''
  Computes the precision@k for the specified values of k
  '''
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk,1,True,True)
  pred = pred.t()
  correct = pred.eq(target.view(1,-1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res
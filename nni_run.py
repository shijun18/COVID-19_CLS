from __future__ import print_function
import numpy as np
from trainer import AverageMeter, accuracy
from run import get_cross_validation

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.transforms as tr

import os
import logging

import nni

import data_utils.transform as tr
from data_utils.data_loader import DataGenerator
from data_utils.csv_reader import csv_reader_single



_logger = logging.getLogger("Covid-19_CLS_pytorch_automl")


train_loader = None
val_loader = None
net = None
criterion = None
optimizer = None
lr_scheduler = None
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

FOLD_NUM = 5

net_name = 'se_r3d_18'
channels = 1
num_classes = 3
input_shape = (64,224,224)
crop = 48
batch_size = 6

train_path = []
val_path = []


def val_on_epoch(epoch):

    net.eval()

    val_loss = AverageMeter()
    val_acc = AverageMeter()

    with torch.no_grad():
        for step, sample in enumerate(val_loader):

            data = sample['image']
            target = sample['label']

            data = data.cuda()
            target = target.cuda()

            output = net(data)
            loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            acc = accuracy(output.data, target)[0]
            val_loss.update(loss.item(), data.size(0))
            val_acc.update(acc.item(), data.size(0))
            torch.cuda.empty_cache()

    return val_loss.avg, val_acc.avg


def train_on_epoch(epoch):

    net.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    for step, sample in enumerate(train_loader):

        data = sample['image']
        target = sample['label']

        data = data.cuda()
        target = target.cuda()

        output = net(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        acc = accuracy(output.data, target)[0]
        train_loss.update(loss.item(), data.size(0))
        train_acc.update(acc.item(), data.size(0))

        torch.cuda.empty_cache()

        print('Train epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'
              .format(epoch, step, loss.item(), acc.item(), optimizer.param_groups[0]['lr']))

    return train_loss.avg, train_acc.avg


def get_net(net_name):
    if net_name == 'r3d_18':
        from model.resnet_3d import r3d_18
        net = r3d_18(input_channels=channels,num_classes=num_classes)
      
    elif net_name == 'r3d_conv_18':
        from model.resnet_conv_3d import r3d_conv_18
        net = r3d_conv_18(input_channels=channels,num_classes=num_classes)

    elif net_name == 'mc3_18':
        from model.resnet_3d import mc3_18
        net = mc3_18(input_channels=channels,num_classes=num_classes)

    elif net_name == 'r2plus1d_18':
        from model.resnet_3d import r2plus1d_18
        net = r2plus1d_18(input_channels=channels,num_classes=num_classes)   
   
    elif net_name == 'se_r3d_18':
        from model.se_resnet_3d import se_r3d_18
        net = se_r3d_18(input_channels=channels,num_classes=num_classes)
    
    elif net_name == 'se_mc3_18':
        from model.se_resnet_3d import se_mc3_18
        net = se_mc3_18(input_channels=channels,num_classes=num_classes)
      
    elif net_name == 'vgg16_3d':
        from model.vgg_3d import vgg16_3d
        net = vgg16_3d(input_channels=channels,num_classes=num_classes)
    
    elif net_name == 'vgg19_3d':
        from model.vgg_3d import vgg19_3d
        net = vgg19_3d(input_channels=channels,num_classes=num_classes)
    return net  

    return net


def prepare(args, train_path, val_path, label_dict):
    global train_loader
    global val_loader
    global net
    global criterion
    global optimizer
    global lr_scheduler

    # Data
    print('==> Preparing data..')
    train_transformer = transforms.Compose([
        tr.CropResize(dim=input_shape,crop=crop),
        tr.RandomTranslationRotationZoom(mode='trz'),
        tr.RandomFlip(mode='hv'),
        tr.To_Tensor(n_class=num_classes)
    ])

    val_transformer = transforms.Compose([
        tr.CropResize(dim=input_shape,crop=crop),
        tr.To_Tensor(n_class=num_classes)
    ])

    train_dataset = DataGenerator(
        train_path, label_dict, transform=train_transformer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_dataset = DataGenerator(
        val_path, label_dict, transform=val_transformer)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    net = get_net(net_name)
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    if args['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            net.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])

    if args['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    if args['lr_scheduler'] == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args['milestones'], gamma=args['gamma'])
    if args['lr_scheduler'] == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                       optimizer, T_max=args['T_max'])        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cur_fold", type=int, default=1)

    args, _ = parser.parse_known_args()

    try:
        RCV_CONFIG = nni.get_next_parameter()
        _logger.debug(RCV_CONFIG)

        csv_path = './converter/shuffle_label.csv'
        label_dict = csv_reader_single(
            csv_path, key_col='id', value_col='label')
        path_list = list(label_dict.keys())

        fold_losses = []

        for cur_fold in range(1, FOLD_NUM+1):
            train_path, val_path = get_cross_validation(
                path_list, FOLD_NUM, cur_fold)
            prepare(RCV_CONFIG, train_path, val_path, label_dict)

            fold_best_val_loss = 1.
            for epoch in range(start_epoch, start_epoch+args.epochs):
                epoch_train_loss, epoch_train_acc = train_on_epoch(epoch)
                epoch_val_loss, epoch_val_acc = val_on_epoch(epoch)

                if lr_scheduler is not None:
                    lr_scheduler.step()

                print('Fold %d | Epoch %d | Val Loss %.5f | Acc %.5f'
                      % (cur_fold, epoch, epoch_val_loss, epoch_val_acc))

                fold_best_val_loss = min(fold_best_val_loss, epoch_val_loss)
                nni.report_intermediate_result(epoch_val_loss)

            fold_losses.append(fold_best_val_loss)
            break
        nni.report_final_result(np.mean(fold_losses))
    except Exception as exception:
        _logger.exception(exception)
        raise
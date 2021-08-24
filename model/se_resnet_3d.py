import sys
sys.path.append("..")
import torch
import torch.nn as nn

from model.resnet_3d import _volume_resnet,Conv3DSimple,BasicStem,Conv3DNoTemporal


class SELayer(nn.Module):
  def __init__(self, channel, reduction=16):
      super(SELayer, self).__init__()
      self.avg_pool = nn.AdaptiveAvgPool3d(1)
      self.fc = nn.Sequential(
          nn.Linear(channel, channel // reduction),
          nn.ReLU(inplace=True),
          nn.Linear(channel // reduction, channel),
          nn.Sigmoid()
      )

  def forward(self, x):
      b, c, _, _,_ = x.size()
      y = self.avg_pool(x).view(b, c)
      y = self.fc(y).view(b, c, 1, 1, 1)
      return x * y.expand_as(x)




class SEBasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, groups=1,
                base_width=64, dilation=1, norm_layer=None,
                *, reduction=16):
      super(SEBasicBlock, self).__init__()
      self.conv1 = conv_builder(inplanes, planes, stride=stride)
      self.bn1 = nn.BatchNorm3d(planes)
      self.relu = nn.ReLU(inplace=True)
      self.conv2 = conv_builder(planes, planes, stride=1)
      self.bn2 = nn.BatchNorm3d(planes)
      self.se = SELayer(planes, reduction)
      self.downsample = downsample
      self.stride = stride

  def forward(self, x):
      residual = x
      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)
      out = self.se(out)

      if self.downsample is not None:
          residual = self.downsample(x)

      out += residual
      out = self.relu(out)

      return out



class SEBottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, groups=1,
                base_width=64, dilation=1, norm_layer=None,
                *, reduction=16):
      super(SEBottleneck, self).__init__()
      self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
      self.bn1 = nn.BatchNorm3d(planes)
      
      self.conv2 = conv_builder(planes, planes,stride=stride)
      self.bn2 = nn.BatchNorm3d(planes)
      
      self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
      self.bn3 = nn.BatchNorm3d(planes * self.expansion)
      
      self.relu = nn.ReLU(inplace=True)
      
      self.se = SELayer(planes * 4, reduction)
      self.downsample = downsample
      self.stride = stride

  def forward(self, x):
      residual = x

      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)
      out = self.relu(out)

      out = self.conv3(out)
      out = self.bn3(out)
      out = self.se(out)

      if self.downsample is not None:
          residual = self.downsample(x)

      out += residual
      out = self.relu(out)

      return out




def se_r3d_18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: SE_R3D-18 network
    """

    return _volume_resnet('se_r3d_18',
                         pretrained, progress,
                         block=SEBasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


def se_r3d_34(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: SE_R3D-18 network
    """

    return _volume_resnet('se_r3d_34',
                         pretrained, progress,
                         block=SEBasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[3, 4, 6, 3],
                         stem=BasicStem, **kwargs)

def se_mc3_18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: SE_MC3-18 network
    """

    return _volume_resnet('se_mc3_18',
                         pretrained, progress,
                         block=SEBasicBlock,
                         conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


def get_parameter_number(net):
  total_num = sum(p.numel() for p in net.parameters())
  trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
  return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
  
  net = se_mc3_18(input_channels=1,num_classes=3)

  print(get_parameter_number(net))
  from torchsummary import summary
  import os 
  os.environ['CUDA_VISIBLE_DEVICES'] = '4'
  net = net.cuda()
  summary(net,input_size=(1,64,224,224),batch_size=1,device='cuda')
  #print(net)
  #modules = net.named_children()
  #for name, module in modules:
  #    print(name,module)
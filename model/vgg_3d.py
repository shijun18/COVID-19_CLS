import torch
import torch.nn as nn

try:
  from torch.hub import load_state_dict_from_url
except ImportError:
  from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['vgg11','vgg13','vgg16','vgg19']


model_urls = {
  'vgg11': None,
  'vgg13': None,
  'vgg16': None,
  'vgg19': None,
}

down_mode = 'M'

cfgs = {
  'A': [32, down_mode, 64, down_mode, 128, 128, down_mode, 256, 256, down_mode, 256, 256, down_mode],
  'B': [32, 32, down_mode, 64, 64, down_mode, 128, 128, down_mode, 256, 256, down_mode, 256, 256, down_mode],
  'D': [32, 32, down_mode, 64, 64, down_mode, 128, 128, 128, down_mode, 256, 256, 256, down_mode, 256, 256, 256, down_mode],
  'E': [32, 32, down_mode, 64, 64, down_mode, 128, 128, 128, 128, down_mode, 256, 256, 256, 256, down_mode, 256, 256, 256, 256, down_mode],
}



class VGG(nn.Module):

  def __init__(self,cfg,num_classes=3,input_channels=1,init_weights=True):
    super(VGG, self).__init__()
    
    self.input_channels = input_channels
    self.features = self._make_layers(cfg)
    self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
    self.classifier = nn.Sequential(
      nn.Linear(256, 256),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(256, 128),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(128, num_classes),
    )
    if init_weights:
      self._initialize_weights()

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
  

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


  def _make_layers(self,cfg):
    layers = []
    in_channels = self.input_channels
    for v in cfg:
      if v == 'M':
        layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
      # 'C': use convolutional layer as downsampling 
      elif v == 'C':
        layers += nn.Conv3d(in_channels,in_channels,kernel_size=3, stride=2, bias=False)
        layers += nn.BatchNorm3d(in_channels)
      else:
        conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
        layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)



""""<https://arxiv.org/pdf/1409.1556.pdf>"""


def _vgg(arch,cfg, pretrained, progress, **kwargs):
  
  model = VGG(cfg=cfgs[cfg], **kwargs)
  if pretrained:
    state_dict = load_state_dict_from_url(model_urls[arch],
                                          progress=progress)
    model.load_state_dict(state_dict)
  return model


def vgg11_3d(pretrained=False, progress=True, **kwargs):
  """VGG 11-layer model (configuration "A") from
  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Args:
      pretrained (bool): If True, returns a model pre-trained
      progress (bool): If True, displays a progress bar of the download to stderr
  """
  return _vgg('vgg11', 'A', pretrained, progress, **kwargs)



def vgg13_3d(pretrained=False, progress=True, **kwargs):
  """VGG 13-layer model (configuration "B")
  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Args:
      pretrained (bool): If True, returns a model pre-trained
      progress (bool): If True, displays a progress bar of the download to stderr
  """
  return _vgg('vgg13', 'B', pretrained, progress, **kwargs)



def vgg16_3d(pretrained=False, progress=True, **kwargs):
  """VGG 16-layer model (configuration "D")
  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Args:
      pretrained (bool): If True, returns a model pre-trained
      progress (bool): If True, displays a progress bar of the download to stderr
  """
  return _vgg('vgg16', 'D',pretrained, progress, **kwargs)




def vgg19_3d(pretrained=False, progress=True, **kwargs):
  """VGG 19-layer model (configuration "E")
  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Args:
      pretrained (bool): If True, returns a model pre-trained
      progress (bool): If True, displays a progress bar of the download to stderr
  """
  return _vgg('vgg19', 'E', pretrained, progress, **kwargs)



if __name__ == "__main__":
  
  net = vgg19_3d(input_channels=1,num_classes=3)

  from torchsummary import summary
  import os 
  os.environ['CUDA_VISIBLE_DEVICES'] = '5'
  net = net.cuda()
  summary(net,input_size=(1,64,224,224),batch_size=1,device='cuda')
  #print(net)
  # modules = net.named_children()
  # for name, module in modules:
  #     print(name,module)
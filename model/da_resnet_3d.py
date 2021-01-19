import sys
sys.path.append("..")
import torch
import torch.nn as nn

try:
  from torch.hub import load_state_dict_from_url
except ImportError:
  from torch.utils.model_zoo import load_url as load_state_dict_from_url

from model.resnet_3d import Conv3DSimple,BasicStem,Conv3DNoTemporal



model_urls = {
  'da_18':None
}


class Conv3DKeepDepth(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DKeepDepth, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=(1, stride, stride),
            padding=(padding, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)




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


# DepthAttention simple version
# class DepthAttention(nn.Module):
#     def __init__(self, channel, depth=64):
#         super(DepthAttention,self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d((depth,1,1))
        

#     def forward(self,x):
#         b,c,d,_,_ = x.size()
#         y = self.avg_pool(x)
#         return x*y.expand_as(x)



class DepthAttention(nn.Module):
    def __init__(self, channel, depth=64):
        super(DepthAttention,self).__init__()
        reduction = channel
        self.avg_pool = nn.AdaptiveAvgPool3d((depth,1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel*depth,(channel*depth) // reduction),
            nn.ReLU(inplace=True),
            nn.Linear((channel*depth)// reduction,channel*depth),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,d,_,_ = x.size()
        y = self.avg_pool(x).view(b,c*d)
        y = self.fc(y).view(b,c,d,1,1)
        return x*y.expand_as(x)


class DABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, depth=64):
        super(DABasicBlock, self).__init__()
        self.conv1 = conv_builder(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_builder(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.da = DepthAttention(planes,depth) 
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.da(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DASEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, depth=64, reduction=16):
        super(DASEBasicBlock, self).__init__()
        self.conv1 = conv_builder(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_builder(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.da = DepthAttention(planes,depth) 
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
        out = self.da(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class DABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, depth=64, reduction=16):
        super(DABottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        
        self.conv2 = conv_builder(planes, planes,stride=stride)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.da = DepthAttention(planes * self.expansion, depth)
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
        out = self.da(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out





class DASEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, depth=64, reduction=16):
        super(DASEBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        
        self.conv2 = conv_builder(planes, planes,stride=stride)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.da = DepthAttention(planes * self.expansion, depth)
        self.se = SELayer(planes * self.expansion, reduction)
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
        out = self.da(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class VolumeResNet(nn.Module):

    def __init__(self, block, conv_makers,layers,depths,stem,num_classes=3,input_channels=1,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 10.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VolumeResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem(input_channels=input_channels)

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1, depth=depths[0])
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2, depth=depths[1])
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2, depth=depths[2])
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2, depth=depths[3])

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DABottleneck) or isinstance(m, DASEBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, depth=64):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample, depth=depth))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder, depth=depth))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _volume_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VolumeResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model





def da_18(pretrained=False, progress=True, **kwargs):

    return _volume_resnet('da_18',
                         pretrained, progress,
                         block=DABasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         depths=[64,32,16,8],
                         stem=BasicStem, **kwargs)


def da_se_18(pretrained=False, progress=True, **kwargs):

    return _volume_resnet('da_se_18',
                         pretrained, progress,
                         block=DASEBasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         depths=[64,32,16,8],
                         stem=BasicStem, **kwargs)


def da_mc3_18(pretrained=False, progress=True, **kwargs):

    return _volume_resnet('da_mc3_18',
                         pretrained, progress,
                         block=DABasicBlock,
                         conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
                         layers=[2, 2, 2, 2],
                         depths=[64,64,64,64],
                         stem=BasicStem, **kwargs)


def da_se_mc3_18(pretrained=False, progress=True, **kwargs):

    return _volume_resnet('da_se_mc3_18',
                         pretrained, progress,
                         block=DASEBasicBlock,
                         conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
                         layers=[2, 2, 2, 2],
                         depths=[64,64,64,64],
                         stem=BasicStem, **kwargs)


def get_parameter_number(net):
  total_num = sum(p.numel() for p in net.parameters())
  trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
  return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
  
#   net = da_mc3_18(input_channels=1,num_classes=3)
  net = da_18(input_channels=1,num_classes=3)

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
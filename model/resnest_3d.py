import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_3d import Conv3DSimple, Conv2Plus1D, Conv3DNoTemporal, BasicStem, R2Plus1dStem

try:
  from torch.hub import load_state_dict_from_url
except ImportError:
  from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['r3d_resnest18', 'mc3_resnest18', 'r2plus1d_resnest10']


model_urls = {
  'r3d_resnest18':None, 
  'mc3_resnest10':None, 
  'r2plus1d_resnest18':None,
}

class SplAtConv3d(nn.Module):
    def __init__(self,in_channels,channels,conv_builder,stride = 1, padding = 0,
            groups = 1,radix = 2,reduction_factor = 4,norm_layer = None,**kwargs):
        super(SplAtConv3d,self).__init__()
        inter_channels = max(in_channels*radix//reduction_factor,32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.use_bn = norm_layer is not None
        midplanes = (in_channels+channels*radix)//2

        self.conv = conv_builder(in_channels,channels * radix,midplanes,stride,
                groups = groups * radix, **kwargs )
        if self.use_bn :
            self.bn0 = norm_layer(channels * radix)

        self.relu = nn.ReLU(inplace = True)
        self.fc1 = nn.Conv3d(channels,inter_channels,1,groups = self.cardinality)
        if self.use_bn :
            self.bn1 = norm_layer(inter_channels)
        
        self.fc2 = nn.Conv3d(inter_channels,channels*radix,1,groups = self.cardinality)
        self.rsoftmax = rSoftMax(radix,groups)

    def forward(self,x):
        x = self.conv(x)  
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)

        batch,rchannel = x.shape[:2]
        if self.radix >1:
            splited = torch.split(x,rchannel//self.radix,dim = 1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool3d(gap,1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch,-1,1,1,1)
        if self.radix > 1:
            attens = torch.split(atten,rchannel//self.radix,dim = 1)  #分成cardinal k块 
            out = sum([att*split for (att,split) in zip(attens,splited)])
        else:
            out = atten * x 

        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self,radix,cardinality):
        super(rSoftMax,self).__init__()
        self.radix = radix
        self.cardinality = cardinality
    
    def forward(self,x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch,self.cardinality,self.radix,-1).transpose(1,2)
            x = F.softmax(x,dim = 1) #每一个分组进行单独进行softmax
            x  = x.reshape(batch,-1)
        else:
            x = torch.sigmoid(x)
        return x

class GlobalAvgPool3d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool3d,self).__init__()

    def forward(self,inputs):
        return nn.functional.adaptive_avg_pool3d(inputs,1).view(inputs.size(0),-1)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,planes,conv_builder,stride = 1,downsample = None,
            radix = 1,cardinality = 1,bottleneck_width = 64,norm_layer = None):
        super(BasicBlock,self).__init__()
        # midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        group_width = int(planes * (bottleneck_width / 64.))*cardinality
        self.radix = radix
        if self.radix >= 1:
            self.conv1 = SplAtConv3d(inplanes,group_width,conv_builder,stride=stride,
                groups=cardinality, radix=radix,norm_layer = norm_layer)
            self.conv2 = SplAtConv3d(group_width,planes,conv_builder,
                groups=cardinality, radix=radix,norm_layer = norm_layer)
        else:
            self.conv1 = conv_builder(inplanes,group_width,(group_width+inplanes)//2,stride = stride,
                groups = cardinality)
            self.bn1 = norm_layer(group_width)
            self.conv2 = conv_builder(group_width,planes,(group_width+planes)//2,
                groups = cardinality)
            self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        
    def forward(self,x):
        residual = x

        out = self.conv1(x)
        if self.radix == 0 :
            out = self.bn1(out)
            out = self.relu(out)
        
        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,conv_builder,stride = 1,downsample = None,
            radix = 1,cardinality = 1,bottleneck_width = 64,norm_layer = None):
        super(Bottleneck,self).__init__()
        # midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        group_width = int(planes * (bottleneck_width / 64.))*cardinality
        self.conv1 = nn.Conv3d(inplanes,group_width,kernel_size = 1, bias = False)
        self.bn1 = norm_layer(group_width)
        self.radix = radix
      
        if radix >= 1 :
            self.conv2 = SplAtConv3d(group_width,group_width,conv_builder,stride=stride,
                groups=cardinality,radix=radix,norm_layer = norm_layer)
        else :
            self.conv2 = conv_builder(group_width,group_width,group_width,stride = stride,
                groups = cardinality)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv3d(group_width,planes*4,kernel_size = 1,bias = False)
        self.bn3 = norm_layer(planes*4)

        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
     
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)
            
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
          
        out += residual
        out = self.relu(out)

        return out

class ResNeSt(nn.Module):
    def __init__(self,block,conv_makers,layers,stem,num_classes = 3,input_channels = 1,
            radix = 1,groups = 1,bottleneck_width = 64,final_drop = 0.0,
            norm_layer = nn.BatchNorm3d):
        super(ResNeSt,self).__init__()
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.radix = radix
        self.inplanes = 64
        # self.norm_layer = norm_layer
        
        self.stem = stem(input_channels,64,norm_layer)

        self.layer1 = self._make_layer(block, conv_makers[0], 64,layers[0],norm_layer = norm_layer)
        self.layer2 = self._make_layer(block, conv_makers[1], 128,layers[1],stride = 2,norm_layer = norm_layer)
        self.layer3 = self._make_layer(block, conv_makers[2], 256,layers[2],stride = 2,norm_layer = norm_layer)
        self.layer4 = self._make_layer(block, conv_makers[3], 512,layers[3],stride = 2,norm_layer = norm_layer)

        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.avgpool = GlobalAvgPool3d()
        self.fc = nn.Linear(512 * block.expansion , num_classes)

    def forward(self,x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self,block,conv_builder,planes,blocks,stride = 1,norm_layer = None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion :
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample,
                radix = self.radix, cardinality = self.cardinality,
                bottleneck_width = self.bottleneck_width, norm_layer = norm_layer ))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder, 
                radix = self.radix,cardinality = self.cardinality,
                bottleneck_width = self.bottleneck_width,norm_layer = norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    
def _volume_resnest(arch,pretrained = False, progress = True, **kwargs):
    model = ResNeSt(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(models_url[arch], progress = progress)
        model.load_state_dict(state_dict)
    return model

def r3d_resnest18(pretrained = False, progress = True, **kwargs):
    return _volume_resnest('r3d_resnest18', pretrained, progress,
            block = BasicBlock, conv_makers = [Conv3DSimple] * 4,
            layers = [2,2,2,2], stem = BasicStem, **kwargs )

def mc_resnest10(pretrained = False, progress = True, **kwargs):
    return _volume_resnest('mc_resnest18', pretrained, progress,
            block = BasicBlock, conv_makers = [Conv3DSimple] + [Conv3DNoTemporal] * 3,
            layers = [1,1,1,1], stem = BasicStem, **kwargs )

def r2plus1d_resnest18(pretrained = False, progress = True, **kwargs):
    return _volume_resnest('r2plus1d_resnest18', pretrained, progress,
                block = BasicBlock, conv_makers = [Conv2Plus1D] * 4,
                layers = [2,2,2,2], stem = R2Plus1dStem, **kwargs)


# if __name__ == "__main__":
  
#   net = mc_resnest18(input_channels=1,num_classes=3,radix = 2)
# #   print(net)

#   from torchsummary import summary
#   import os 
#   os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#   net = net.cuda()
#   summary(net,input_size=(1,64,224,224),batch_size=1,device='cuda')
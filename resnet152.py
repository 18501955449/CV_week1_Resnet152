import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# class SELayer(nn.Module):
#     '''测试SE模块'''
#     def __init__(self,channel,reduction=16):
#         super(SELayer,self).__init__()
          #整个空间特征编码为一个全局特征
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
          #通过两个全连接层，第一个进行降维作用，降低模型复杂度，提高泛化能力
          #第二个全连接层恢复到原始维度
#         self.fc = nn.Sequential(
#             nn.Linear(channel,channel//reduction,bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel//reduction,channel,bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self,x):
#         b,c,_,_ = x.size()
#         y = self.avg_pool(x).view(b,c)
#         y = self.fc(y).view(b, c, 1, 1)
          #最后将学习到的各个channel的权重系数乘以原始特征，算是一种attention机制
#         return x * y.expand_as(x)

model_urls = {'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}
def conv3_3(in_planes,out_planes,stride = 1,groups = 1,dilation = 1):
    '''3*3卷积,其中bias=False是因为加入BN层，没有必要加入偏置项'''
    return nn.Conv2d(in_planes,out_planes,kernel_size = 3,stride = stride,
                     padding = dilation,groups = groups,bias = False,dilation = dilation)

def conv1_1(in_planes,out_planes,stride = 1):
    '''1*1卷积'''
    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias = False)

class Bottleneck(nn.Module):
    '''resnet单元，大于50层的加入了1*1卷积降维升维的部分,主要目的是降低参数量'''
    expansion = 4
    def __init__(self,inplanes,planes,stride=1,downsample=None,groups = 1,
                 base_width=64,dilation=1,norm_layer = None):
        super(Bottleneck,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #这里默认卷积分为1组64个channel，还可以分多个groups，每个groups设置width_per_group个channel
        width = int(planes*(base_width/64.))*groups
        self.conv1 =conv1_1(inplanes,width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3_3(width,width,stride,groups,dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1_1(width,planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        #加入SENet的位置
        #self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        identity = x
        #1*1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #3*3卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #1*1卷积
        out = self.conv3(out)
        out = self.bn3(out)
        # 加入SENet的位置
        # out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=1000,groups = 1,width_per_group=64,
                 norm_layer=None):
        super(ResNet,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,self.inplanes,kernel_size=7,stride=2,padding = 3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self,block,planes,blocks,stride=1,dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            #这里是为了转换为相同的维度能够进行shortcut，即尺寸和维度都要相同
            downsample = nn.Sequential(
                conv1_1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        #以下这个地方挺绕，先添加第一个block，然后再改变输入维度为planes*4，然后再添加剩余block
        #其中self.inplanes=64开始，每个block结束通道变为palnes*4
        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample,self.groups,
                            self.base_width,previous_dilation,norm_layer))
        self.inplanes = planes*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

def _resnet(arch,block,layers,pretrained,progress,**kwargs):
    model = ResNet(block,layers,**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],progress=progress)
        model.load_state_dict(state_dict)
    return model
def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,**kwargs)

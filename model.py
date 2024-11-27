import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)

class middle_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(middle_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.middle = model_t

    def forward(self, x):
        x = self.middle.conv1(x)
        x = self.middle.bn1(x)
        x = self.middle.relu(x)
        x = self.middle.maxpool(x)
        return x


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class embed_net(nn.Module):
    def __init__(self,  class_num, dataset, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.middle_module = middle_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        self.dataset = dataset

        if self.dataset == 'regdb':
            pool_dim = 1024
        elif self.dataset == 'sysu':
            pool_dim = 2048
        elif self.dataset == 'llcm':
            pool_dim = 2048

        self.l2norm = Normalize(2)
        self.bottleneck1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck1.bias.requires_grad_(False)  # no shift
        self.bottleneck1.apply(weights_init_kaiming)
        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier1.apply(weights_init_classifier)

        self.bottleneck2 = nn.BatchNorm1d(pool_dim)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2.apply(weights_init_classifier)

        self.bottleneck3 = nn.BatchNorm1d(pool_dim)
        self.bottleneck3.bias.requires_grad_(False)  # no shift
        self.bottleneck3.apply(weights_init_kaiming)
        self.classifier3 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier3.apply(weights_init_classifier)

        self.bottleneckg = nn.BatchNorm1d(pool_dim)
        self.bottleneckg.bias.requires_grad_(False)  # no shift
        self.bottleneckg.apply(weights_init_kaiming)
        self.classifierg = nn.Linear(pool_dim, class_num, bias=False)
        self.classifierg.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block
        x = self.base_resnet.base.layer1(x)
        x = self.base_resnet.base.layer2(x)
        x = self.base_resnet.base.layer3(x)
        if self.dataset == 'regdb':
            x = x
        elif self.dataset == 'sysu':
            x = self.base_resnet.base.layer4(x)
        elif self.dataset == 'llcm':
            x = self.base_resnet.base.layer4(x)
        
        x1, x2, x3 = torch.chunk(x, 3, 2)
        xp1 = self.avgpool(x1)
        xp1 = xp1.view(xp1.size(0), xp1.size(1))
        xp2 = self.avgpool(x2)
        xp2 = xp2.view(xp2.size(0), xp2.size(1))
        xp3 = self.avgpool(x3)
        xp3 = xp3.view(xp3.size(0), xp3.size(1))
        xpg = self.avgpool(x)
        xpg = xpg.view(xpg.size(0), xpg.size(1))
        

        ft1 = self.bottleneck1(xp1)
        ft2 = self.bottleneck2(xp2)
        ft3 = self.bottleneck3(xp3)
        ftg = self.bottleneckg(xpg)

        if self.training:
            return xp1, xp2, xp3, xpg, self.classifier1(ft1), self.classifier2(ft2), self.classifier3(ft3), self.classifierg(ftg)
        else:
            return self.l2norm(torch.cat((xp1, xp2, xp3, xpg), 1)), self.l2norm(torch.cat((ft1, ft2, ft3, ftg), 1))
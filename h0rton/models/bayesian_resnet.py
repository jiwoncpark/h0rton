import torch
import torchvision.models as models
from torchvision.models.resnet import conv1x1, BasicBlock
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet44', 'resnet56']

class BayesianBasicBlock(BasicBlock):
    """Basic block of ResNet BNN with architectural modifications from the torchvision implementation

    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, dropout_rate=0.0):
        super(BayesianBasicBlock, self).__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer)
        self.dropout_rate = dropout_rate

    def forward(self, x):   
        identity = x
        out = F.dropout2d(x, p=self.dropout_rate)   
        out = self.conv1(out)  
        out = self.bn1(out) 
        out = self.relu(out)    

        out = F.dropout2d(out, p=self.dropout_rate) 
        out = self.conv2(out)   
        out = self.bn2(out) 

        if self.downsample is not None: 
            identity = self.downsample(x) 

        out += identity 
        out = self.relu(out)   
        return out

class BayesianResNet(models.ResNet):
    """ResNet BNN with architectural modifications from the torchvision implementation

    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dropout_rate=0.0):
        self.dropout_rate = dropout_rate
        self.inplanes = 64
        super(BayesianResNet, self).__init__(block, layers, num_classes, zero_init_residual,
                 groups, width_per_group, replace_stride_with_dilation,
                 norm_layer)
        # Override first conv layer 
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove fourth layer so number of filters in FC should be 256, not 512
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Override _make_layer to pass in dropout_rate
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        # Note 4th layer is never used
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, dropout_rate=self.dropout_rate))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()
        x = F.dropout2d(x, p=self.dropout_rate) # F not NN b/c activated during eval
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        x = self.avgpool(x)
        x = F.dropout(x, p=self.dropout_rate)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def _resnet(arch, block, layers, progress, **kwargs):
    model = BayesianResNet(block, layers, **kwargs)
    return model

def resnet44(progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    
    """
    return _resnet('resnet44', BayesianBasicBlock, [7, 7, 7, 1], progress,
                   **kwargs)

def resnet56(progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    
    """
    return _resnet('resnet56', BayesianBasicBlock, [9, 9, 9, 1], progress,
                   **kwargs)
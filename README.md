# pytorch-MobileNet
Simple Code Implementation of ["MobileNet"](https://arxiv.org/abs/1704.04861) architecture using PyTorch.

![](https://github.com/hoya012/pytorch-MobileNet/blob/master/assets/mobilenet.PNG)

For simplicity, i write codes in `ipynb`. So, you can easliy test my code.

*Last update : 2018/12/19*

## Contributor
* hoya012

## Requirements
Python 3.5
```
numpy
matplotlib
torch=1.0.0
torchvision
```

## Usage
You only run `MobileNet-pytorch.ipynb`.
For test, i used `CIFAR-10` Dataset and resize image scale from 32x32 to 224x224.
If you want to use own dataset, you can simply resize images.

## depthwise convolution and other blocks impelemtation.
In MobileNet, there are many depthwise convolution operation. This is my simple implemenatation.

### depthwise convolution operation
```
class depthwise_conv(nn.Module):
    def __init__(self, nin, kernel_size, padding, bias=False, stride=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        return out
```

### depthwise block
```
class dw_block(nn.Module):
    def __init__(self, nin, kernel_size, padding=1, bias=False, stride=1):
        super(dw_block, self).__init__()
        self.dw_block = nn.Sequential(
            depthwise_conv(nin, kernel_size, padding, bias, stride),
            nn.BatchNorm2d(nin),
            nn.ReLU(True)
        )
    def forward(self, x):
        out = self.dw_block(x)
        return out
```

### 1x1 block
```
class one_by_one_block(nn.Module):
    def __init__(self, nin, nout, padding=1, bias=False, stride=1):
        super(one_by_one_block, self).__init__()
        self.one_by_one_block = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=1, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(nout),
            nn.ReLU(True)
        )
    def forward(self, x):
        out = self.one_by_one_block(x)
        return out
```


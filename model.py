import torch
from torch import nn, Tensor
from torchvision import transforms,models


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Classifier(nn.Module):
    def __init__(
            self,
            n_classes: int,
            in_planes: int,
            out_planes: int,
            stride: int = 1
    ) -> None:
        super(Classifier, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(out_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(out_planes, out_planes, stride)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_planes, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class MobileNetv2(nn.Module):
    def __init__(self, n_classes=[3, 1]):
        super(MobileNetv2, self).__init__()

        self.backbone = models.mobilenet_v2(pretrained=True).features
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier1 = Classifier(n_classes[0], 1280, 640, 1)
        self.classifier2 = Classifier(n_classes[1], 1280, 640, 1)

        for m in self.classifier1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.classifier2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        y1 = self.classifier1(x)
        y2 = self.classifier2(x)
        return {'class': y1, 'def': y2}


class Densenet121(nn.Module):
    def __init__(self, n_classes=[3, 1]):
        super(Densenet121, self).__init__()

        self.backbone = models.densenet121(pretrained=True).features
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier1 = Classifier(n_classes[0], 1024, 512, 1)
        self.classifier2 = Classifier(n_classes[1], 1024, 512, 1)

        for m in self.classifier1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.classifier2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        y1 = self.classifier1(x)
        y2 = self.classifier2(x)
        return {'class': y1, 'def': y2}


class Resnet50(nn.Module):
    def __init__(self, n_classes=[3, 1]):
        super(Resnet50, self).__init__()

        self.backbone = models.resnet50(pretrained=True)
        self.backbone2 = list(self.backbone.children())[:-1]
        self.backbone = nn.Sequential(*self.backbone2)
        
        a = 0
        for i in self.backbone.children():
            a += 1
            if a < 7:
                for param in i.parameters():
                    param.requires_grad = False

        self.classifier1 = Classifier(n_classes[0], 2048, 1024, 1)
        self.classifier2 = Classifier(n_classes[1], 2048, 1024, 1)

        for m in self.classifier1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.classifier2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        y1 = self.classifier1(x)
        y2 = self.classifier2(x)
        return {'class': y1, 'def': y2}


class inceptionV3(nn.Module):
    def __init__(self, n_classes=[20, 129]):
        super(inceptionV3, self).__init__()
        self.backbone=models.inception_v3(pretrained=True)
        self.backbone2=list(self.backbone.children())[:-1]
        self.backbone=nn.Sequential(*self.backbone2)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier1 = Classifier(n_classes[0], 196, 1024, 1)
        self.classifier2 = Classifier(n_classes[1], 196, 1024, 1)

        for m in self.classifier1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.classifier2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        y1 = self.classifier1(x)
        y2 = self.classifier2(x)
        return {'class': y1, 'def': y2}


class Densenet169(nn.Module):
    def __init__(self, n_classes=[3, 1]):
        super(Densenet169, self).__init__()

        self.backbone = models.densenet169(pretrained=True).features
        a = 0
        for i in self.backbone.children():
            a += 1
            if a < 9:
                for param in i.parameters():
                    param.requires_grad = False
        self.classifier1 = Classifier(n_classes[0], 1664, 832, 1)
        self.classifier2 = Classifier(n_classes[1], 1664, 832, 1)

        for m in self.classifier1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.classifier2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        y1 = self.classifier1(x)
        y2 = self.classifier2(x)
        return {'class': y1, 'def': y2}

class Densenet201(nn.Module):
    def __init__(self, n_classes=[3, 1]):
        super(Densenet201, self).__init__()

        self.backbone = models.densenet201(pretrained=False).features
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier1 = Classifier(n_classes[0], 1920, 960, 1)
        self.classifier2 = Classifier(n_classes[1], 1920, 960, 1)

        for m in self.classifier1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.classifier2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        y1 = self.classifier1(x)
        y2 = self.classifier2(x)
        return {'class': y1, 'def': y2}



class Densenet201x(nn.Module):
    def __init__(self, n_classes=[3, 1]):
        super(Densenet201x, self).__init__()

        self.backbone = models.densenet201(pretrained=False).features
        a = 0
        for i in self.backbone.children():
            a += 1
            if a < 9:
                for param in i.parameters():
                    param.requires_grad = False

        self.classifier1 = Classifier(n_classes[0], 1920, 960, 1)
        self.classifier2 = Classifier(n_classes[1], 1920, 960, 1)

        for m in self.classifier1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.classifier2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        y1 = self.classifier1(x)
        y2 = self.classifier2(x)
        return {'class': y1, 'def': y2}


class MobileNetv2x(nn.Module):
    def __init__(self, n_classes=[3, 1]):
        super(MobileNetv2x, self).__init__()

        self.backbone = models.mobilenet_v2(pretrained=True).features
        a = 0
        for i in self.backbone.children():
            a += 1
            if a < 0:
                for param in i.parameters():
                    param.requires_grad = False

        self.classifier1 = Classifier(n_classes[0], 1280, 640, 1)
        self.classifier2 = Classifier(n_classes[1], 1280, 640, 1)

        for m in self.classifier1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.classifier2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        y1 = self.classifier1(x)
        y2 = self.classifier2(x)
        return {'class': y1, 'def': y2}


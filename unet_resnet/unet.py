from torch import nn
import torch
from ccsdg.models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F

# SaveFeatures 类用于保存模型中间层的特征。它包含一个静态变量 features，用于存储特征值。
# 在初始化时，通过注册一个前向钩子（hook）函数 hook_fn，
# 将特征值保存到 features 变量中。remove 方法用于移除前向钩子。

class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()

# UnetBlock 类是 U-Net 模型中的一个基本块。
# 它接收一个上采样输入 up_p，一个特征输入 x_p，
# 并输出一个合并后的特征。在初始化时，
# 它包含一个上采样卷积层、一个特征卷积层和一个批归一化层。在前向传播中，
# 通过上采样卷积和特征卷积对输入进行卷积操作，
# 然后将它们在通道维度上拼接，最后通过批归一化和 ReLU 激活函数输出合并后的特征。

class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class UNet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        fea = x
        output = self.up5(x)

        return output

    def close(self):
        for sf in self.sfs: sf.remove()

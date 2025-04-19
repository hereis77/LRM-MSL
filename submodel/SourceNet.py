import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.cbam_attention import CBAMBlock
from torchinfo import summary
from thop import profile
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class _sourcenet(nn.Module):
    def __init__(self, num_classes=2, input_channels=2, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512, 512]

        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = ResVGGBlock(input_channels, nb_filter[0], nb_filter[0], cbam=True)
        self.conv1_0 = ResVGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], cbam=True)
        self.conv2_0 = ResVGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], cbam=True)
        self.conv3_0 = ResVGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], cbam=True)
        self.conv4_0 = ResVGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], cbam=True)
        self.conv5_0 = ResVGGBlock(nb_filter[4], nb_filter[5], nb_filter[5], cbam=True)

        self.fc_layers = nn.Sequential(
            nn.Linear(1504, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        # print(x0_0.size())
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print(x1_0.size())
        x2_0 = self.conv2_0(self.pool(x1_0))
        # print(x2_0.size())
        x3_0 = self.conv3_0(self.pool(x2_0))
        # print(x3_0.size())
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print(x4_0.size())
        x5_0 = self.conv5_0(self.pool(x4_0))
        # print(x5_0.size())

        x5 = F.adaptive_avg_pool2d(x5_0, 1).view(x5_0.size(0), -1)
        # print(x5.size())
        x4 = F.adaptive_avg_pool2d(x4_0, 1).view(x4_0.size(0), -1)
        # print(x4.size())
        x3 = F.adaptive_avg_pool2d(x3_0, 1).view(x3_0.size(0), -1)
        # print(x3.size())
        x2 = F.adaptive_avg_pool2d(x2_0, 1).view(x3_0.size(0), -1)
        # print(x2.size())
        x1 = F.adaptive_avg_pool2d(x1_0, 1).view(x3_0.size(0), -1)
        # print(x1.size())
        x0 = F.adaptive_avg_pool2d(x0_0, 1).view(x3_0.size(0), -1)
        # print(x0.size())
        x = torch.cat((x5,x4, x3, x2, x1, x0), dim=1)
        # print(x.size())
        output = self.fc_layers(x)
        return output

class ResVGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, cbam=True):
        super().__init__()
        self.cbam = cbam
        self.ca = CoordAtt(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else None


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.res_conv:
            identity = self.res_conv(identity)

        out += identity
        out = self.ca(out)
        out = self.relu(out)

        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


if __name__ == '__main__':
    model = _sourcenet().to(device)
    input = torch.rand(1, 2, 256, 256).to(device)
    output = model(input)
    summary(model, input_size=(1, 2, 256, 256))
    print(input.size(), output.size())  # 应该是 (16, 2)
    flops, params = profile(model, inputs=(input,))
    print('the flops is {}G,the params is {}M'.format(round(flops / (10 ** 9), 2),
                                                      round(params / (10 ** 6), 2)))

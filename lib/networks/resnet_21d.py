"""
Modify the original file to make the class support feature extraction
"""
import torch.nn as nn
import torch


class Block21d(nn.Module):

    def __init__(self, in_channels, out_channels, s_stride=1, t_stride=1, downsample=None):
        super(Block21d, self).__init__()

        m_resume1 = int(in_channels * out_channels * 3 * 3 * 3 / (in_channels * 3 * 3 + out_channels * 3))
        self.conv1_1 = nn.Conv3d(in_channels=in_channels, out_channels=m_resume1,
                                 kernel_size=(1, 3, 3), stride=(1, s_stride, s_stride), padding=(0, 1, 1),
                                 bias=True)
        self.bn1_1 = nn.BatchNorm3d(m_resume1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv3d(in_channels=m_resume1, out_channels=out_channels,
                                 kernel_size=(3, 1, 1), stride=(t_stride, 1, 1), padding=(1, 0, 0),
                                 bias=True)
        self.bn1_2 = nn.BatchNorm3d(out_channels)

        m_resume2 = int(out_channels * out_channels * 3 * 3 * 3 / (out_channels * 3 * 3 + out_channels * 3))
        self.conv2_1 = nn.Conv3d(in_channels=out_channels, out_channels=m_resume2,
                                 kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                                 bias=True)
        self.bn2_1 = nn.BatchNorm3d(m_resume2)
        self.conv2_2 = nn.Conv3d(in_channels=m_resume2, out_channels=out_channels,
                                 kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0),
                                 bias=True)
        self.bn2_2 = nn.BatchNorm3d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1_1(x)
        out = self.bn1_1(out)
        out = self.relu(out)

        out = self.conv1_2(out)
        out = self.bn1_2(out)
        out = self.relu(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.relu(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet21D(nn.Module):

    def __init__(self, block, layers, num_classes=400, feat=False, lite=False, **kwargs):
        if not isinstance(block, list):
            block = [block] * 4
        else:
            assert (len(block)) == 4, "Block number must be 4 for ResNet-Stype networks."

        super(ResNet21D, self).__init__()

        self.feat = feat

        self.conv1_1 = nn.Conv3d(3, 83, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=True)
        self.bn1_1 = nn.BatchNorm3d(83)
        self.relu = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv3d(83, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=True)
        self.bn1_2 = nn.BatchNorm3d(64)

        self.in_channels = 64

        self.layer2 = self._make_layer(type_block=block[0], out_channels=64, num_blocks=layers[0],
                                       t_stride=(1 if not lite else 2))
        self.layer3 = self._make_layer(type_block=block[1], out_channels=128, num_blocks=layers[1],
                                       s_stride=2, t_stride=2)
        self.layer4 = self._make_layer(type_block=block[2], out_channels=256, num_blocks=layers[2],
                                       s_stride=2, t_stride=2)
        self.layer5 = self._make_layer(type_block=block[3], out_channels=512, num_blocks=layers[3],
                                       s_stride=2, t_stride=2)

        t_length = 16
        self.avgpool = nn.AvgPool3d(kernel_size=((int(t_length / 8) if not lite else 1), 7, 7), stride=1)

        self.feat_dim = 512
        if not feat:
            self.fc = nn.Linear(512, num_classes)

        # initialization
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, type_block, out_channels, num_blocks, s_stride=1, t_stride=1):
        downsample = None
        if s_stride != 1 or t_stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels,
                          kernel_size=1, stride=(t_stride, s_stride, s_stride), bias=True),
                nn.BatchNorm3d(out_channels)
            )

        layers = [type_block(self.in_channels, out_channels, s_stride=s_stride, t_stride=t_stride, downsample=downsample)]
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(type_block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if not self.feat:
            x = self.fc(x)

        return x


def resnet18_21d(pretrained=False, feat=False, **kwargs):
    model = ResNet21D(Block21d, [2, 2, 2, 2], feat=feat, **kwargs)

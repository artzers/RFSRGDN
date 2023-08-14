import torch
import torch.nn as nn
from torch.nn import functional as F
from Util import PixelUpsampler3D


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.PReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

class Upsample(nn.Module):
    """
    upsample, concat and conv
    """

    def __init__(self, in_channel, inter_channel, out_channel):
        super(Upsample, self).__init__()
        self.up = nn.Sequential(
            ConvBlock(in_channel, inter_channel),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )
        self.conv = ConvBlock(2 * inter_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        out = torch.cat((x1, x2), dim=1)
        out = self.conv(out)
        return out

class Upsample3(nn.Module):
    """
    upsample, concat and conv
    """

    def __init__(self, in_channel, out_channel):
        super(Upsample3, self).__init__()
        self.up = nn.Sequential(
            ConvBlock(in_channel, out_channel),
            nn.Upsample(scale_factor=(3,3,3),mode='trilinear')
        )

    def forward(self, x1):
        x1 = self.up(x1)
        return x1


class AttentionGate(nn.Module):
    """
    filter the features propagated through the skip connections
    """

    def __init__(self, in_channel, gating_channel, inter_channel):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv3d(gating_channel, inter_channel, kernel_size=1)
        self.W_x = nn.Conv3d(in_channel, inter_channel, kernel_size=2, stride=2)
        self.relu = nn.PReLU()
        self.psi = nn.Conv3d(inter_channel, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x, g):
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        out = self.relu(g_conv + x_conv)
        out = self.sig(self.psi(out))
        out = F.upsample(out, size=x.size()[2:], mode='trilinear')
        out = x * out
        return out


class AttentionUNet(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel, num_class, filters=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()

        f1, f2, f3, f4 = filters

        self.down1 = ConvBlock(in_channel, f1)

        self.down2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            ConvBlock(f1, f2)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            ConvBlock(f2, f3)
        )

        self.down4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            ConvBlock(f3, f4)
        )

        self.ag1 = AttentionGate(f3, f4, f3)
        self.ag2 = AttentionGate(f2, f2, f2)
        self.ag3 = AttentionGate(f1, f1, f1)

        self.up1 = Upsample(f4, f3, f2)
        self.up2 = Upsample(f2, f2, f1)
        self.up3 = Upsample(f1, f1, f1)

        out_feats = 3 * 3 * 3* f1
        tail = []

        self.skip = nn.Sequential(
            ConvBlock(in_channel, f1),
            nn.Upsample(scale_factor=(3,3,3), mode='trilinear')
            #PixelUpsampler3D((3, 3, 3))
        )
        # tail.append(
        #     wn(nn.Conv3d(f1, f1, 3, padding=3 // 2)))
        tail.append(Upsample3(f1, f1))
        #tail.append(PixelUpsampler3D((3, 3, 3)))
        self.tail = nn.Sequential(*tail)
        self.tail1 = nn.Conv3d(2*f1, f1, kernel_size=3, padding=1)
        self.act1 = nn.PReLU()
        self.tail2 = nn.Conv3d(f1, num_class, kernel_size=3, padding=1)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.bias, 0.0)
                nn.init.normal_(m.weight, 1.0, 0.02)
            elif isinstance(m, nn.GroupNorm):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        ag1 = self.ag1(down3, down4)
        up1 = self.up1(down4, ag1)
        ag2 = self.ag2(down2, up1)
        up2 = self.up2(up1, ag2)
        ag3 = self.ag3(down1, up2)
        up3 = self.up3(up2, ag3)
        up3 = self.tail(up3)
        up3 = torch.cat([self.skip(x), up3], dim=1)
        up3 = self.tail1(up3)
        up3 = self.act1(up3)
        up3 = self.tail2(up3)
        return up3




    #downsample generator
class DegradeNet(nn.Module):
    def __init__(self):
        super(DegradeNet, self).__init__()
        self.nFeat = 48
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act = nn.LeakyReLU#(inplace=True)

        layer1 = [
            nn.Conv3d(1,self.nFeat,3,3//2),
            act(inplace=True)
            ]
        layer2=[
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            # wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            # act(inplace=True),
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            nn.MaxPool3d(3, padding=1),
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3,  padding=3 // 2)),
            act(inplace=True),
            #nn.MaxPool3d(3, padding=1),
            wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            nn.Conv3d(self.nFeat, 1, 3, padding=3 // 2)
        ]

        self.head = nn.Sequential(*layer1)
        self.body = nn.Sequential(*layer2)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.bias, 0.0)
                nn.init.normal_(m.weight, 1.0, 0.02)
            elif isinstance(m, nn.GroupNorm):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)


    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return x


#discriminator low image
class LowDiscriminator(nn.Module):
    def __init__(self):
        super(LowDiscriminator, self).__init__()
        self.nFeat = 64
        self.output_shape = (1,1,1,1)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act = nn.LeakyReLU
        layer1 = [
            #wn(nn.Conv3d(1,self.nFeat,3,3//2)),
            (nn.Conv3d(1, self.nFeat, 3, 3 // 2)),
            act(inplace=True),
        ]
        layer2 = [
            #wn(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            # (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            # act(inplace=True),
            nn.MaxPool3d(3, stride=2),
            #(nn.Conv3d(self.nFeat, self.nFeat, 3, stride=2, padding=3 // 2)),
            #act(inplace=True),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            # (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            # act(inplace=True),
            nn.MaxPool3d(3, stride=2),
            #(nn.Conv3d(self.nFeat, self.nFeat, 3, stride=2, padding=3 // 2)),
            #act(inplace=True),
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
            # (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            # act(inplace=True),
            nn.MaxPool3d(3, stride=2),
            #(nn.Conv3d(self.nFeat, self.nFeat, 3, stride=2, padding=3 // 2)),
            #act(inplace=True),
            (nn.Conv3d(self.nFeat, 1, 3, padding=3 // 2)),
            #act(inplace=True),
            #nn.Conv3d(self.nFeat, 1, 2)
        ]
        self.head = nn.Sequential(*layer1)
        self.body = nn.Sequential(*layer2)
        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.bias, 0.0)
                nn.init.normal_(m.weight, 1.0, 0.02)
            elif isinstance(m, nn.GroupNorm):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)


    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return x

#discriminator high image
class HighDiscriminator(nn.Module):
    def __init__(self):
        super(HighDiscriminator, self).__init__()

        self.nFeat = 48
        self.output_shape = (1,1,1,1)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        act = nn.LeakyReLU
        layer1 = [
            #wn(nn.Conv3d(1, self.nFeat, 3, padding=3 // 2)),
            (nn.Conv3d(1, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True),
        ]
        layer2 = [
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        self.mp1 = nn.MaxPool3d(2, stride=2)
        layer3 = [
            (nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        layer4 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        layer5 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        self.mp2 = nn.MaxPool3d(2, stride=2)

        layer6 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        layer7 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(True)]
        self.mp3 = nn.MaxPool3d(2, stride=2)
        layer8 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        layer9 = [(nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2)),
            act(inplace=True)]
        self.mp4 = nn.MaxPool3d(2, stride=2)
        layer10 = [nn.Conv3d(self.nFeat,self.nFeat, 3, padding=3 // 2),
            act(inplace=True)]
        # layer11 = [nn.Conv3d(self.nFeat, self.nFeat, 3, padding=3 // 2),
        #     act(inplace=True)]
        self.mp4 = nn.MaxPool3d(2, stride=2)
        self.tail = nn.Conv3d(self.nFeat, 1, 3, padding=3 // 2)

        self.head = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.layer5 = nn.Sequential(*layer5)
        self.layer6 = nn.Sequential(*layer6)
        self.layer7 = nn.Sequential(*layer7)
        self.layer8 = nn.Sequential(*layer8)
        self.layer9 = nn.Sequential(*layer9)
        self.layer10 = nn.Sequential(*layer10)
        # self.layer11 = nn.Sequential(*layer11)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.bias, 0.0)
                nn.init.normal_(m.weight, 1.0, 0.02)
            elif isinstance(m, nn.GroupNorm):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)


    def forward(self, x):
        x = self.head(x)
        x = self.layer2(x)
        x = self.mp1(x)
        x = self.layer3(x)
        f1 = self.layer4(x)
        x = self.mp2(f1)
        x = self.layer5(x)
        f2 = self.layer6(x)
        x = self.mp3(f2)
        x = self.layer7(x)
        f3 = self.layer8(x)
        x = self.mp4(f3)
        x = self.layer9(x)
        x = self.layer10(x)
        # x = self.layer11(x)
        return x,f1,f2,f3


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.utils import LF_rgb2ycbcr
DispRange = 4


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        feaC = 16
        channel = 160
        mindisp, maxdisp = -DispRange, DispRange
        angRes = args.angRes_in
        self.angRes = angRes
        self.init_feature = nn.Sequential(
            nn.Conv2d(1, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feaC, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            nn.Conv2d(feaC,  feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feaC, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            )

        self.build_costvolume = BuildCostVolume(feaC, channel, angRes, mindisp, maxdisp)

        self.aggregation = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.LeakyReLU(0.1, inplace=True),
            ResB3D(channel),
            ResB3D(channel),
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.regression = Regression(mindisp, maxdisp)

    def forward(self, lf, info):
        lf = torch.mean(lf, dim=1, keepdim=True)

        [b, c, u, v, h, w] = lf.size()
        lf = rearrange(lf, 'b c u v h w -> b c (u h) (v w)')

        x = SAI2MacPI(lf, self.angRes)
        init_feat = self.init_feature(x)
        cost = self.build_costvolume(init_feat)
        cost = self.aggregation(cost)
        init_disp = self.regression(cost.squeeze(1))

        out = {}
        out['Disp'] = init_disp
        return out


class BuildCostVolume(nn.Module):
    def __init__(self, channel_in, channel_out, angRes, mindisp, maxdisp):
        super(BuildCostVolume, self).__init__()
        self.DSAFE = nn.Conv2d(channel_in, channel_out, angRes, stride=angRes, padding=0, bias=False)
        self.angRes = angRes
        self.mindisp = mindisp
        self.maxdisp = maxdisp

    def forward(self, x):
        cost_list = []
        for d in range(self.mindisp, self.maxdisp + 1):
            if d < 0:
                dilat = int(abs(d) * self.angRes + 1)
                pad = int(0.5 * self.angRes * (self.angRes - 1) * abs(d))
            if d == 0:
                dilat = 1
                pad = 0
            if d > 0:
                dilat = int(abs(d) * self.angRes - 1)
                pad = int(0.5 * self.angRes * (self.angRes - 1) * abs(d) - self.angRes + 1)
            cost = F.conv2d(x, weight=self.DSAFE.weight, stride=self.angRes, dilation=dilat, padding=pad)
            cost_list.append(cost)
        cost_volume = torch.stack(cost_list, dim=2)

        return cost_volume


class Regression(nn.Module):
    def __init__(self, mindisp, maxdisp):
        super(Regression, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.maxdisp = maxdisp
        self.mindisp = mindisp

    def forward(self, cost):
        score = self.softmax(cost)              # B, D, H, W
        temp = torch.zeros(score.shape).to(score.device)            # B, D, H, W
        for d in range(self.maxdisp - self.mindisp + 1):
            temp[:, d, :, :] = score[:, d, :, :] * (self.mindisp + d)
        disp = torch.sum(temp, dim=1, keepdim=True)     # B, 1, H, W

        return disp


class SpaResB(nn.Module):
    def __init__(self, channels, angRes):
        super(SpaResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        buffer = self.body(x)
        return buffer + x


class ResB3D(nn.Module):
    def __init__(self, channels):
        super(ResB3D, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x):
        buffer = self.body(x)
        return buffer + x


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, criterion_data=None):
        loss = self.criterion_Loss(out['Disp'], HR[:, 0:1, ])

        return loss
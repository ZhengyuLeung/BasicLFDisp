import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from option import args
import importlib


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes = args.angRes_in
        disp = 9
        self.feature_extraction = feature_extraction()
        self.get_CostVolume = get_CostVolume(disp)

        self.channel_attention = channel_attention(disp, self.angRes)
        self.basic = basic(self.angRes)

    def forward(self, lf, info):
        [b, c, u, v, h, w] = lf.size()

        """ features """
        lf = rearrange(lf, 'b c u v h w -> b c (u v) h w')
        input = self.feature_extraction(lf)
        # input = rearrange(input, 'b c (u v) h w -> b c u v h w', u=u, v=v)

        """ cost volume """
        cv = self.get_CostVolume(input)
        cv = rearrange(cv, 'b d c a h w -> b (c a) d h w')

        """ channel attention """
        cv, attention = self.channel_attention(cv)

        """ cost volume regression """
        cost = self.basic(cv).squeeze(1)
        pred = F.softmax(cost, dim=1)
        disparity_values = torch.linspace(-4, 4, 9).view(1, -1, 1, 1).to(pred.device)
        disp = torch.sum(pred * disparity_values, dim=1, keepdim=True)

        out = {}
        out['Disp'] = disp
        return out


class basic(nn.Module):
    def __init__(self, angRes):
        super(basic, self).__init__()
        feature = 2 * 75
        self.conv0 = nn.Sequential(
            conv3x3_bn(in_channels=4*angRes**2, out_channels=feature),
            nn.ReLU(True),
            conv3x3_bn(in_channels=feature, out_channels=feature),
            nn.ReLU(True),
        )
        self.conv1 = nn.Sequential(
            conv3x3_bn(in_channels=feature, out_channels=feature),
            nn.ReLU(True),
            conv3x3_bn(in_channels=feature, out_channels=feature),
        )
        self.conv2 = nn.Sequential(
            conv3x3_bn(in_channels=feature, out_channels=feature),
            nn.ReLU(True),
            conv3x3_bn(in_channels=feature, out_channels=feature),
        )
        self.classify = nn.Sequential(
            conv3x3_bn(in_channels=feature, out_channels=feature),
            nn.ReLU(True),
            nn.Conv3d(feature, 1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, cost_volume):
        cost0 = self.conv0(cost_volume)

        cost0 = self.conv1(cost0) + cost0
        cost0 = self.conv2(cost0) + cost0
        cost = self.classify(cost0)

        return cost


class channel_attention(nn.Module):
    def __init__(self, disp, angRes):
        super(channel_attention, self).__init__()
        self.disp = disp
        self.angRes = angRes
        self.conv = nn.Sequential(
            nn.Conv3d(4*angRes**2, 170, kernel_size=1),
            nn.ReLU(True),
            nn.Conv3d(170, 15, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, cost_volume):
        x = F.adaptive_avg_pool3d(cost_volume, output_size=(1, 1, 1))
        x = self.conv(x)
        x = torch.cat([x[:, 0:5], x[:, 1:2], x[:, 5:9], x[:, 2:3],
                       x[:, 6:7], x[:, 9:12], x[:, 3:4], x[:, 7:8],
                       x[:, 10:11], x[:, 12:14], x[:, 4:5], x[:, 8:9],
                       x[:, 11:12], x[:, 13:15]], dim=1)
        x = x.view(-1, 5, 5)
        b, _, _ = x.size()
        ####9X9padding updated####
        x = F.pad(x, (0, self.angRes - 5), mode='reflect')
        x = x.permute(0, 2, 1)
        x = F.pad(x, (0, self.angRes - 5), mode='reflect')
        x = x.permute(0, 2, 1)
        attention = x.reshape(-1, self.angRes**2, 1, 1, 1)
        attention = repeat(attention, 'b A 1 1 1 -> b (4 A) 1 1 1')
        return attention * cost_volume, attention


class to_3d(nn.Module):
    def __init__(self, angRes):
        super(to_3d, self).__init__()
        self.angRes = angRes
        feature = 4 * angRes
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Conv3d(feature, feature//2, kernel_size=1),
            nn.ReLU(True),
            nn.Conv3d(feature//2, 3, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(feature, feature // 2, kernel_size=1),
            nn.ReLU(True),
            nn.Conv3d(feature // 2, 3, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv3 = nn.Sequential(
            conv3x3_bn(in_channels=feature, out_channels=feature),
            nn.ReLU(True),
            conv3x3_bn(in_channels=feature, out_channels=feature//2),
            nn.ReLU(True),
            conv3x3_bn(in_channels=feature//2, out_channels=feature//2),
            nn.ReLU(True),
            conv3x3_bn(in_channels=feature//2, out_channels=feature//4),
            nn.ReLU(True),
            conv3x3_bn(in_channels=feature//4, out_channels=1),
            nn.ReLU(True),
        )

    def forward(self, cost_volume):
        channel_h = self.conv1(cost_volume)
        channel_h = repeat(channel_h[:, 0:1,], 'b 1 1 1 1 -> b n 1 1 1', n=self.angRes//2)
        channel_h = torch.cat([repeat(channel_h[:, 0:1, ], 'b 1 1 1 1 -> b n 1 1 1', n=self.angRes // 2),
                               channel_h[:, 1:2, ],
                               repeat(channel_h[:, 2:3, ], 'b 1 1 1 1 -> b n 1 1 1', n=self.angRes // 2), ], dim=1)
        channel_h = torch.cat([channel_h, channel_h, channel_h, channel_h], dim=1)
        cv_h_tmp = channel_h * cost_volume

        cv_h_tmp = self.conv2(cv_h_tmp)
        attention_h = torch.cat([repeat(cv_h_tmp[:, 0:1, ], 'b 1 a h w -> b n a h w', n=self.angRes // 2),
                               cv_h_tmp[:, 1:2, ],
                               repeat(cv_h_tmp[:, 2:3, ], 'b 1 a h w -> b n a h w', n=self.angRes // 2), ], dim=1)
        attention_h = torch.cat([attention_h, attention_h, attention_h, attention_h], dim=1)
        cv_h_multi = attention_h * cost_volume

        cost0 = self.conv3(cv_h_multi)
        cost0 = cost0.squeeze(1)
        return cost0, cv_h_multi


class get_CostVolume(nn.Module):
    def __init__(self, disp):
        super(get_CostVolume, self).__init__()
        self.disp = disp // 2

    def forward(self, inputs):
        [b, c, a, h, w] = inputs.size()
        disparity_costs = []
        for d in range(-self.disp, self.disp + 1):
            cost = self.SheardEPI(inputs.clone(), d)
            disparity_costs.append(cost)

        cost_volume = torch.stack(disparity_costs, dim=1)
        return cost_volume

    def SheardEPI(self, LF, shift):
        [_, _, A, _, _] = LF.size()
        for a in range(A):
            t1 = int(shift * (a - A // 2))
            t2 = int(shift * (a - A // 2))
            LF[:, :, a, :, :] = torch.roll(LF[:, :, a, :, :], shifts=t1, dims=2)
            LF[:, :, a, :, :] = torch.roll(LF[:, :, a, :, :], shifts=t2, dims=3)
            if t1 < 0:
                LF[:, :, a, t1::, :] = 0
            elif t1 > 0:
                LF[:, :, a, 0:t1, :] = 0
            if t2 < 0:
                LF[:, :, a, :, t2::] = 0
            elif t2 > 0:
                LF[:, :, a, :, 0:t2] = 0
        return LF


class conv3x3_bn(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(conv3x3_bn, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1,dilation,dilation), bias=False),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, seq):
        seq = self.body(seq)
        return seq


class conv1x1_bn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv1x1_bn, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, seq):
        seq = self.body(seq)
        return seq


class BasicBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3_bn(channels, channels, dilation=1),
            nn.ReLU(True),
        )
        self.conv2 = conv3x3_bn(channels, channels, dilation=1)

    def forward(self, buffer):
        buffer = self.conv1(buffer)
        buffer = self.conv2(buffer) + buffer
        return buffer


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.firstconv = nn.Sequential(
            conv3x3_bn(in_channels=3, out_channels=4),
            nn.ReLU(True),
            conv3x3_bn(in_channels=4, out_channels=4),
            nn.ReLU(True),
        )
        self.layer1 = nn.Sequential(
            BasicBlock(channels=4),
            BasicBlock(channels=4),
            BasicBlock(channels=4)
        )
        self.layer2 = nn.Sequential(
            conv1x1_bn(in_channels=4, out_channels=8),
            BasicBlock(channels=8),
            BasicBlock(channels=8),
            BasicBlock(channels=8),
            BasicBlock(channels=8),
            BasicBlock(channels=8),
            BasicBlock(channels=8),
            BasicBlock(channels=8),
            BasicBlock(channels=8),
        )
        self.layer3 = nn.Sequential(
            conv1x1_bn(in_channels=8, out_channels=16),
            BasicBlock(channels=16),
            BasicBlock(channels=16),
        )
        self.layer4 = nn.Sequential(
            conv1x1_bn(in_channels=16, out_channels=16),
            BasicBlock(channels=16, dilation=2),
            BasicBlock(channels=16, dilation=2),
        )
        self.branch1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            conv3x3_bn(in_channels=16, out_channels=4),
            nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            conv3x3_bn(in_channels=4, out_channels=4),
            nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 8, 8), stride=(1, 8, 8)),
            conv3x3_bn(in_channels=4, out_channels=4),
            nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 16, 16), stride=(1, 16, 16)),
            conv3x3_bn(in_channels=4, out_channels=4),
            nn.ReLU(True),
        )
        self.lastconv = nn.Sequential(
            conv3x3_bn(in_channels=40, out_channels=16),
            nn.ReLU(True),
            nn.Conv3d(16, 4, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=False),
        )

    def forward(self, x):
        [b, c, u, h, w] = x.size()
        buffer = self.firstconv(x)
        layer1 = self.layer1(buffer)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        branch1 = self.branch1(layer4)
        branch1 = rearrange(branch1, 'b c u h w -> (b u) c h w')
        branch1 = F.interpolate(branch1, scale_factor=2, mode='bilinear')
        branch1 = rearrange(branch1, '(b u) c h w -> b c u h w', u=u)

        branch2 = self.branch2(branch1)
        branch2 = rearrange(branch2, 'b c u h w -> (b u) c h w')
        branch2 = F.interpolate(branch2, scale_factor=4, mode='bilinear')
        branch2 = rearrange(branch2, '(b u) c h w -> b c u h w', u=u)

        branch3 = self.branch3(branch2)
        branch3 = rearrange(branch3, 'b c u h w -> (b u) c h w')
        branch3 = F.interpolate(branch3, scale_factor=8, mode='bilinear')
        branch3 = rearrange(branch3, '(b u) c h w -> b c u h w', u=u)

        branch4 = self.branch4(branch3)
        branch4 = rearrange(branch4, 'b c u h w -> (b u) c h w')
        branch4 = F.interpolate(branch4, scale_factor=16, mode='bilinear')
        branch4 = rearrange(branch4, '(b u) c h w -> b c u h w', u=u)

        output_feature = torch.cat([layer2, layer4, branch4, branch3, branch2, branch1], dim=1)
        lastconv = self.lastconv(output_feature)
        return lastconv


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, criterion_data=None):
        mask = (1 - HR[:, 1:2, ])
        loss = self.criterion_Loss(out['Disp'] * mask, HR[:, 0:1, ] * mask)

        return loss

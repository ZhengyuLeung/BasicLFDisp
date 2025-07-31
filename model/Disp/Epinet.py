import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.utils import LF_rgb2ycbcr
from option import args
import importlib


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        angRes = args.angRes_in
        self.angRes = angRes
        filt_num = 70
        self.mid_0d = layer1_multistream(angRes, filt_num)
        self.mid_90d = layer1_multistream(angRes, filt_num)
        self.mid_45d = layer1_multistream(angRes, filt_num)
        self.mid_M45d = layer1_multistream(angRes, filt_num)
        self.layer2_merged = layer2_merged(filt_num*4, conv_depth=7)
        self.layer3_last = layer3_last(filt_num*4)

    def forward(self, lf, info):
        [B, c, u, v, h, w] = lf.size()

        ''' 4-Input '''
        input_stack_0d = lf[:, :, self.angRes // 2, :, :, :]
        input_stack_90d = lf[:, :, :, self.angRes // 2, :, :]

        lf = rearrange(lf, 'b c u v h w -> b c (u v) h w')
        input_stack_45d = []
        for i in range(0, self.angRes ** 2, self.angRes + 1):
            input_stack_45d.append(lf[:, :, i, :, :])
        input_stack_45d = torch.stack(input_stack_45d, dim=2)
        input_stack_M45d = []
        for i in range(self.angRes - 1, self.angRes ** 2 - 1, self.angRes - 1):
            input_stack_M45d.append(lf[:, :, i, :, :])
        input_stack_M45d = torch.stack(input_stack_M45d, dim=2)

        ''' 4-Stream layer : Conv - Relu - Conv - BN - Relu '''
        mid_0d = self.mid_0d(input_stack_0d)
        mid_90d = self.mid_90d(input_stack_90d)
        mid_45d = self.mid_45d(input_stack_45d)
        mid_M45d = self.mid_M45d(input_stack_M45d)

        ''' Merge layers '''
        mid_merged = torch.cat([mid_0d, mid_90d, mid_45d, mid_M45d], dim=1)

        ''' Merged layer : Conv - Relu - Conv - BN - Relu '''
        mid_merged_ = self.layer2_merged(mid_merged)

        ''' Last Conv layer : Conv - Relu - Conv '''
        output = self.layer3_last(mid_merged_)

        out = {}
        out['Disp'] = output
        return out


class layer3_last(nn.Module):
    def __init__(self, filt_num):
        super(layer3_last, self).__init__()
        body = []
        body.append(nn.Conv2d(filt_num, filt_num, kernel_size=3, padding=1))
        body.append(nn.ReLU(True))
        body.append(nn.Conv2d(filt_num, 1, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

    def forward(self, buffer):
        buffer = self.body(buffer)
        return buffer


class layer2_merged(nn.Module):
    def __init__(self, filt_num, conv_depth):
        super(layer2_merged, self).__init__()
        body = []
        for _ in range(conv_depth):
            body.append(nn.Conv2d(filt_num, filt_num, kernel_size=3, padding=1))
            body.append(nn.ReLU(True))
            body.append(nn.Conv2d(filt_num, filt_num, kernel_size=3, padding=1))
            body.append(nn.BatchNorm2d(filt_num))
            body.append(nn.ReLU(True))
        self.body = nn.Sequential(*body)

    def forward(self, buffer):
        buffer = self.body(buffer)
        return buffer


class layer1_multistream(nn.Module):
    def __init__(self, angRes, filt_num):
        super(layer1_multistream, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(angRes*3, filt_num, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(filt_num),
            nn.ReLU(True),
            nn.Conv2d(filt_num, filt_num, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(filt_num),
            nn.ReLU(True),
            nn.Conv2d(filt_num, filt_num, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(filt_num),
            nn.ReLU(True),
        )

    def forward(self, buffer):
        [b, c, a, h, w] = buffer.size()
        buffer = rearrange(buffer, 'b c a h w -> b (c a) h w')
        buffer = self.body(buffer)
        return buffer


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, criterion_data=None):
        loss = self.criterion_Loss(out['Disp'], HR[:, 0:1, ])

        return loss
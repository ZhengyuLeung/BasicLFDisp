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
        input_size2 = 48
        assert input_size2 == args.patch_for_train
        filt_num = 32
        self.ORM1 = ORM(input_size2, 3, filt_num)
        self.ORM2 = ORM(input_size2, 3, filt_num)
        self.layer2_1 = layer2(input_size2+3, filt_num)
        self.layer2_2 = layer2(input_size2+3, filt_num)
        self.layer_mid = layer_mid(filt_num)
        self.ORM3 = ORM(input_size2, filt_num, filt_num)
        self.ORM4 = ORM(input_size2, filt_num, filt_num)
        self.res_block_1 = res_block(angRes, input_size2+filt_num, filt_num)
        self.res_block_2 = res_block(angRes, input_size2+filt_num, filt_num)
        self.merge = nn.Sequential(
            nn.Conv2d(filt_num*2, filt_num, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(filt_num, filt_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(filt_num),
            nn.ReLU(True),
            nn.Conv2d(filt_num, filt_num*2, kernel_size=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(filt_num*2, 1, kernel_size=1, padding=0),
        )

    def forward(self, lf, info):
        [B, c, u, v, h, w] = lf.size()
        input_x = lf[:, :, :, self.angRes // 2, :, :]
        input_y = lf[:, :, self.angRes // 2, :, :, :]

        input_x = rearrange(input_x, 'b c u h w -> (b w) c u h')
        input_y = rearrange(input_y, 'b c v h w -> (b h) c v w')

        mid_merged_uv1 = self.ORM1(input_x)
        mid_merged_uv2 = self.ORM2(input_y)

        mid_input_x = self.layer2_1(mid_merged_uv1)
        mid_input_y = self.layer2_2(mid_merged_uv2)

        mid_input_x = self.layer_mid(mid_input_x)
        mid_input_y = self.layer_mid(mid_input_y)

        mid_uv1 = self.ORM3(mid_input_x)
        mid_uv2 = self.ORM4(mid_input_y)

        mid_input_x_orm = self.res_block_1(mid_uv1)
        mid_input_y_orm = self.res_block_2(mid_uv2)

        mid_input_x_orm = rearrange(mid_input_x_orm, '(b w) c 1 h -> b c h w', w=w)
        mid_input_y_orm = rearrange(mid_input_y_orm, '(b h) c 1 w -> b c h w', h=h)

        mid_merged = torch.cat([mid_input_x_orm, mid_input_y_orm], dim=1)
        output = self.merge(mid_merged)

        out = {}
        out['Disp'] = output
        return out


class res_block(nn.Module):
    def __init__(self, angRes, filt_num_in, filt_num):
        super(res_block, self).__init__()
        self.init = nn.Sequential(
            nn.Conv2d(filt_num_in, filt_num, kernel_size=(1, 1), padding=(0, 0)),
            nn.ReLU(True),
        )
        body = []
        self.num_layer = angRes//2
        for _ in range(self.num_layer):
            body.append(nn.Sequential(
                nn.Conv2d(filt_num, filt_num, kernel_size=(1, 3), padding=(0, 1)),
                nn.ReLU(True),
                nn.Conv2d(filt_num, filt_num, kernel_size=(3, 3), padding=(0, 1)),
                nn.BatchNorm2d(filt_num),
                nn.ReLU(True),
            ))
        self.body = nn.Sequential(*body)

    def forward(self, input_x):
        input_x = self.init(input_x)
        for index in range(self.num_layer):
            input_x = self.body[index](input_x) + input_x[:, :, 1:-1, :]
        return input_x


class layer_mid(nn.Module):
    def __init__(self, filt_num):
        super(layer_mid, self).__init__()
        body = []
        for _ in range(3):
            body.append(nn.Conv2d(filt_num, filt_num, kernel_size=(1, 3), padding=(0, 1)))
            body.append(nn.ReLU(True))
            body.append(nn.Conv2d(filt_num, filt_num, kernel_size=(1, 3), padding=(0, 1)))
            body.append(nn.BatchNorm2d(filt_num))
            body.append(nn.ReLU(True))
        self.body = nn.Sequential(*body)

    def forward(self, input_x):
        input_x = self.body(input_x)
        return input_x


class layer2(nn.Module):
    def __init__(self, input_size2, filt_num):
        super(layer2, self).__init__()
        self.input_size2 = input_size2
        body = [
            nn.Conv2d(input_size2, filt_num, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(filt_num, filt_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(filt_num),
            nn.ReLU(True),
        ]
        for _ in range(3):
            body.append(nn.Conv2d(filt_num, filt_num, kernel_size=3, padding=1))
            body.append(nn.ReLU(True))
            body.append(nn.Conv2d(filt_num, filt_num, kernel_size=3, padding=1))
            body.append(nn.BatchNorm2d(filt_num))
            body.append(nn.ReLU(True))
        self.body = nn.Sequential(*body)

    def forward(self, input_x):
        input_x = self.body(input_x)
        return input_x


class ORM(nn.Module):
    def __init__(self, input_size2, filt_num_in, filt_num):
        super(ORM, self).__init__()
        self.input_size2 = input_size2
        self.ORM_1x1_1_input_x = nn.Conv2d(filt_num_in, filt_num, kernel_size=(1, 1))
        self.ORM_1x1_2_input_x = nn.Conv2d(filt_num_in, filt_num, kernel_size=(1, 1))
        self.Activation = nn.ReLU(True)

    def forward(self, input_x):
        [b, _, v, w] = input_x.size()
        mid_merged_u1 = self.ORM_1x1_1_input_x(input_x)
        mid_merged_v1 = self.ORM_1x1_2_input_x(input_x)
        mid_merged_uv1 = torch.einsum('bcm, bcn -> bmn', mid_merged_u1.view(b, -1, v*w),
                                      mid_merged_v1.view(b, -1, v*w))
        mid_merged_uv1 = mid_merged_uv1.view(b, v, w, v*w)
        mid_merged_uv1 = mid_merged_uv1[:, v//2, :, :]
        mid_merged_uv1 = rearrange(mid_merged_uv1, 'b ww (v w) -> b ww v w', v=v)
        mid_merged_uv1 = self.Activation(mid_merged_uv1)
        mid_merged_uv1 = torch.cat([input_x, mid_merged_uv1], dim=1)

        return mid_merged_uv1


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, criterion_data=None):
        loss = self.criterion_Loss(out['Disp'], HR[:, 0:1, ])

        return loss


if __name__ == "__main__":
    args.model_name = 'EPI_ORM'
    args.angRes_in = 5
    args.patch_for_train = 64

    args.patch_size_for_test = 64

    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    MODEL_PATH = args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args).to(device)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of Parameters: %.4fM' % (total / 1e6))

    input = torch.randn([2, 3, args.angRes_in, args.angRes_in, 64, 64]).to(device)
    GPU_cost = torch.cuda.memory_allocated(0)
    out = net(input, [])
    GPU_cost1 = (torch.cuda.memory_allocated(0) - GPU_cost) / 1024 / 1024 / 1024  # GB
    print('   GPU consumption: %.4fGB' % (GPU_cost1))
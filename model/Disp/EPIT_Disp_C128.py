import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from option import args
import importlib
import matplotlib.pyplot as plt
DISP_RANGE = 10


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = 64
        self.channels = channels
        token_dim = channels
        self.angRes = args.angRes_in

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(3, channels//2, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels//2, channels//2, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels//2, channels//2, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels//2, channels//2, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ################ Alternate AngTrans & SpaTrans ################
        self.num_block = 4
        self.altblock = nn.Sequential(
            AltFilter(channels//2, token_dim, patch_size=1),
            AltFilter(channels//2, token_dim, patch_size=1),
            AltFilter(channels//2, token_dim, patch_size=1),
            AltFilter(channels//2, token_dim, patch_size=1, last_layer=True),
        )

        self.build_costvolume = BuildCostVolume()
        self.aggregation = nn.Sequential(
            nn.Conv3d(2*self.angRes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            Conv3dFilter(channels),
            Conv3dFilter(channels),
            Conv3dFilter(channels),
            Conv3dFilter(channels),
            nn.Conv3d(channels, 1, kernel_size=3, padding=1, bias=False)
        )
        self.regression = Regression()

    def forward(self, lf, info=None):
        [b, c, u, v, h, w] = lf.size()

        # Initial Feature Extraction
        x = rearrange(lf, 'b c u v h w -> b c (u v) h w')
        buffer = self.conv_init0(x)
        buffer = self.conv_init(buffer) + buffer  # [B, C, A^2, h, w]
        buffer = rearrange(buffer, 'b c (u v) h w -> b c u v h w', u=self.angRes, v=self.angRes)

        # Deep Spatial-Angular Correlation Learning
        for index in range(self.num_block):
            buffer, attn_map_uh, attn_map_vw = self.altblock[index](buffer)
        # attn_map_uh, attn_map_vw  # [b, a, d, h, w]

        # Constructing Cost Volume
        cost_volume = self.build_costvolume(attn_map_uh, attn_map_vw)  # b c d h w

        # Aggregation and Regression
        cost_volume = rearrange(cost_volume, 'b n a d h w -> b (n a) d h w')
        cost_volume = self.aggregation(cost_volume).squeeze(1)
        disp = self.regression(cost_volume)

        out = {}
        out['Disp'] = disp
        return out


class BuildCostVolume(nn.Module):
    def __init__(self):
        super(BuildCostVolume, self).__init__()
        self.optimal_transport_uh = OptimalTransport()
        self.optimal_transport_vw = OptimalTransport()

    @staticmethod
    def Shear_Cost_by_h(attn_map_uh):
        [b, a, d, h, w] = attn_map_uh.size()
        for index in range(h):
            shifts = h // 2 - index
            attn_map_uh[:, :, :, index, :] = torch.roll(attn_map_uh[:, :, :, index, :], shifts=shifts, dims=2)
            if shifts < 0:
                attn_map_uh[:, :, shifts::, index, :] = 0
            elif shifts > 0:
                attn_map_uh[:, :, 0:shifts, index, :] = 0
        return attn_map_uh

    @staticmethod
    def Reduce_Disp_by_a(attn_map):
        [b, n, a, d, h, w] = attn_map.size()
        attn_map = rearrange(attn_map, 'b n a d h w -> (b n h w) a d')
        attn_map_new = attn_map.new_zeros(b*n*h*w, a, DISP_RANGE*2+1)
        for index in range(a):
            a_delta = max(abs(index - a//2), 1)
            attn_map_new[:, index, :] = F.adaptive_avg_pool1d(
                attn_map[:, index, d//2-DISP_RANGE*a_delta:d//2+DISP_RANGE*a_delta+1], DISP_RANGE*2+1)
        attn_map_new = rearrange(attn_map_new, '(b n h w) a d -> b n a d h w', b=b, n=n, h=h, w=w)
        return attn_map_new

    def forward(self, attn_map_uh, attn_map_vw):
        [b, a, d, h, w] = attn_map_uh.size()
        attn_map_uh = self.Shear_Cost_by_h(attn_map_uh)  # [b, a, d, h, w]
        attn_map_vw = self.Shear_Cost_by_h(attn_map_vw.transpose(-1, -2)).transpose(-1, -2)
        attn_map = torch.stack([attn_map_uh, attn_map_vw], dim=1)  # [b, 2, a, d, h, w]
        if DISP_RANGE*(a//2)*2+1 > d:
            pad = (DISP_RANGE*(a//2)*2 - d) // 2 + 1
            temp = attn_map.new_zeros(b, 2, a, pad, h, w).float()
            # temp = temp.masked_fill(temp == 0, float('-inf'))
            attn_map = torch.cat([temp, attn_map, temp], dim=3)
        #     d = d + 6
        # cost_volume = attn_map[:, :, :, d // 2 - DISP_RANGE:d // 2 + DISP_RANGE + 1, :, :]
        cost_volume = self.Reduce_Disp_by_a(attn_map)

        # normalize attention to 0-1
        # if softmax
        # attn_map = attn_map.masked_fill(attn_map == 0, float('-inf'))
        # cost_volume = F.softmax(attn_map, dim=3)  # [b, 2n, a, d, h, w]
        # if optimal transport
        # attn_map_uh = rearrange(attn_map_uh, 'b a d h w -> (b a) d h w')
        # attn_map_vw = rearrange(attn_map_vw, 'b a d h w -> (b a) d h w')
        # attn_ot_uh = self.optimal_transport_uh(attn_map_uh)
        # attn_ot_vw = self.optimal_transport_vw(attn_map_vw)
        # attn_ot_uh = rearrange(attn_ot_uh, '(b a) d h w -> b a d h w', b=b)
        # attn_ot_vw = rearrange(attn_ot_vw, '(b a) d h w -> b a d h w', b=b)
        # attn_map = torch.stack([attn_ot_uh, attn_ot_vw], dim=1)  # [b, 2, a, d, h, w]
        # # attn_map = rearrange(attn_map, '')
        return cost_volume


class OptimalTransport(nn.Module):
    def __init__(self):
        super(OptimalTransport, self).__init__()
        self.phi = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # dustbin cost

    def _sinkhorn(self, attn, log_mu, log_nu, iters):
        """
        Sinkhorn Normalization in Log-space as matrix scaling problem.
        Regularization strength is set to 1 to avoid manual checking for numerical issues
        Adapted from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork)

        :param attn: input attention weight, [N,H,W+1,W+1]
        :param log_mu: marginal distribution of left image, [N,H,W+1]
        :param log_nu: marginal distribution of right image, [N,H,W+1]
        :param iters: number of iterations
        :return: updated attention weight
        """

        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for idx in range(iters):
            # scale v first then u to ensure row sum is 1, col sum slightly larger than 1
            v = log_nu - torch.logsumexp(attn + u.unsqueeze(3), dim=2)
            u = log_mu - torch.logsumexp(attn + v.unsqueeze(2), dim=3)

        return attn + u.unsqueeze(3) + v.unsqueeze(2)

    def forward(self, attn):
        [ba, d, h, w] = attn.size()
        attn = rearrange(attn, 'ba d h w -> ba h w d')

        # set marginal to be uniform distribution
        marginal = torch.cat([torch.ones([w]), torch.tensor([w]).float()]) / (2 * w)
        log_mu = marginal.log().to(attn.device).expand(ba, h, w + 1)
        log_nu = marginal.log().to(attn.device).expand(ba, h, w + 1)

        # add dustbins
        similarity_matrix = torch.cat([attn, self.phi.expand(ba, h, w, 1).to(attn.device)], -1)
        similarity_matrix = torch.cat([similarity_matrix, self.phi.expand(ba, h, 1, w + 1).to(attn.device)], -2)

        # sinkhorn
        attn_ot = self._sinkhorn(similarity_matrix, log_mu, log_nu, iters=10)

        # convert back from log space, recover probabilities by normalization 2W
        attn_ot = (attn_ot + torch.log(torch.tensor([2.0 * w]).to(attn.device))).exp()

        return attn_ot[:, :, :-1, :-1]


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

    def forward(self, cost):
        [b, d, h, w] = cost.size()
        score = F.softmax(cost, dim=1)  # b, c, a, d, h, w
        disparity_values = torch.linspace(-DISP_RANGE, DISP_RANGE, 2 * DISP_RANGE + 1).view(1, -1, 1, 1).to(cost.device)
        disparity = torch.sum(score * disparity_values, dim=1, keepdim=True)  # [b a h w]

        return disparity


class Conv2dFilter(nn.Module):
    def __init__(self, channels):
        super(Conv2dFilter, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, buffer):
        buffer = self.conv(buffer) + buffer
        buffer = self.conv(buffer) + buffer
        return buffer


class Conv3dFilter(nn.Module):
    def __init__(self, channels):
        super(Conv3dFilter, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, buffer):
        buffer = self.conv(buffer) + buffer
        return buffer


class BasicTrans(nn.Module):
    def __init__(self, channels, token_dim, num_heads=1, dropout=0.):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, token_dim, bias=False)
        self.norm = nn.LayerNorm(token_dim)
        self.num_heads = num_heads

        self.qk_ff = nn.Linear(token_dim, 2*token_dim, bias=False)
        self.v_ff = nn.Linear(token_dim, token_dim, bias=False)
        self.out_ff = nn.Linear(token_dim, token_dim, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(token_dim, channels, bias=False)

    def forward(self, epi_token):
        epi_token = self.linear_in(epi_token)

        # generate query and key
        epi_token_norm = self.norm(epi_token)
        qk = self.qk_ff(epi_token_norm)
        qk = rearrange(qk, 'tgt_len B (Nh c) -> (B Nh) tgt_len c', Nh=self.num_heads)
        qk = qk / math.sqrt(qk.size(2) // 2)
        [query, key] = torch.chunk(qk, 2, dim=2)

        # generate value
        v = self.v_ff(epi_token)
        v = rearrange(v, 'tgt_len B (Nh c) -> (B Nh) tgt_len c', Nh=self.num_heads)
        value = v / math.sqrt(v.size(2))

        # query x key -> attention map, (B, Nt, E) x (B, Ns, E) -> (B, Nt, Ns)
        attn = torch.einsum('b t c, b s c -> b t s', query, key)
        attn_raw = rearrange(attn, '(B Nh) Nt Ns -> B Nh Nt Ns', Nh=self.num_heads).mean(dim=1)
        attn = self.dropout(self.softmax(attn))
        attn_output_weights = rearrange(attn, '(B Nh) Nt Ns -> B Nh Nt Ns', Nh=self.num_heads).mean(dim=1)

        # atten x value -> output, (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.einsum('b t s, b s c -> b t c', attn, value)
        output = rearrange(output, '(B Nh) tgt_len c -> tgt_len B (Nh c)', Nh=self.num_heads)
        epi_token = self.out_ff(output) + epi_token

        # feed forward
        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)

        return epi_token, attn_output_weights, attn_raw


class AltFilter(nn.Module):
    def __init__(self, channels, token_dim, patch_size, last_layer=False):
        super(AltFilter, self).__init__()
        self.p = patch_size
        self.last_layer = last_layer
        self.epi_trans = BasicTrans(channels * patch_size ** 2, token_dim)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, buffer):
        shortcut = buffer
        [b, _, u, v, h, w] = buffer.size()
        if self.last_layer:
            buffer = rearrange(buffer, 'b c u v h w -> b c (u v) h w')
            buffer = self.conv(buffer)
            buffer = rearrange(buffer, 'b c (u v) h w -> b c u v h w', u=u, v=v) + shortcut

            # Horizontal
            buffer_uh = rearrange(buffer, 'b c u v (h c_h) (w c_w) -> (u h) (b v w) (c c_h c_w)', c_h=self.p, c_w=self.p)
            [_, _, attn_map_uh] = self.epi_trans(buffer_uh)

            # Vertical
            buffer_vw = rearrange(buffer, 'b c u v (h c_h) (w c_w) -> (v w) (b u h) (c c_h c_w)', c_h=self.p, c_w=self.p)
            [_, _, attn_map_vw] = self.epi_trans(buffer_vw)

            # Constructing Cost Volume
            attn_map_uh = rearrange(attn_map_uh, '(b v w) (u h) (uu hh) -> b uu hh u v h w', b=b, u=u, v=v, uu=u)
            attn_map_vw = rearrange(attn_map_vw, '(b u h) (v w) (vv ww) -> b vv ww u v h w', b=b, u=u, v=v, vv=v)
            attn_map_uh = attn_map_uh[:, :, :, u//2, v//2, :, :]
            attn_map_vw = attn_map_vw[:, :, :, u//2, v//2, :, :]
        else:
            # Horizontal
            buffer = rearrange(buffer, 'b c u v (h c_h) (w c_w) -> h (b u v w) (c c_h c_w)', c_h=self.p, c_w=self.p)
            [buffer, _, _] = self.epi_trans(buffer)
            buffer = rearrange(buffer, 'h (b u v w) (c c_h c_w) -> b c (u v) (h c_h) (w c_w)', b=b, u=u, v=v,
                               c_h=self.p, c_w=self.p)
            buffer = self.conv(buffer)
            buffer = rearrange(buffer, 'b c (u v) h w -> b c u v h w', u=u, v=v)
            buffer = buffer + shortcut

            # Vertical
            buffer = rearrange(buffer, 'b c u v (h c_h) (w c_w) -> w (b u v h) (c c_h c_w)', c_h=self.p, c_w=self.p)
            [buffer, _, _] = self.epi_trans(buffer)
            buffer = rearrange(buffer, 'w (b u v h) (c c_h c_w) -> b c (u v) (w c_w) (h c_h)', b=b, u=u, v=v,
                               c_h=self.p, c_w=self.p)
            buffer = self.conv(buffer)
            buffer = rearrange(buffer, 'b c (u v) w h -> b c u v h w', u=u, v=v)
            buffer = buffer + shortcut

            attn_map_uh, attn_map_vw = None, None

        return buffer, attn_map_uh, attn_map_vw


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


if __name__ == "__main__":
    args.model_name = 'EPIT_Disp_optimal'
    args.angRes_in = 3
    args.patch_for_train = 32

    args.patch_size_for_test = 32

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
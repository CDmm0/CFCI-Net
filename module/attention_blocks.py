import torch
import torch.nn as nn
import torch.nn.functional as F


norm_dict = {'BATCH': nn.BatchNorm3d, 'INSTANCE': nn.InstanceNorm3d, 'GROUP': nn.GroupNorm}


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky=True, norm='BATCH', activation=True):
        super().__init__()
        # determine basic attributes
        self.norm_type = norm
        self.activation = activation
        self.leaky = leaky
        padding = (kernel_size - 1) // 2

        # activation, support PReLU and common ReLU
        if self.leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

        # instantiate layers
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        norm_layer = norm_dict[norm]
        if norm in ['BATCH', 'INSTANCE']:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = norm_layer(8, in_channels)

    def basic_forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.act(x)
        return x

    def group_forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

    def forward(self, x):
        if self.norm_type in ['BATCH', 'INSTANCE']:
            return self.basic_forward(x)
        else:
            return self.group_forward(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        if leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, out_channels, 3, stride, leaky, norm, True)
        self.conv2 = ConvNorm(out_channels, out_channels, 3, 1, leaky, norm, False)

        self.identity_mapping = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False)

        self.need_map = in_channels != out_channels or stride != 1

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.need_map:
            identity = self.identity_mapping(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        return out


class ResBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        middle_channels = in_channels // 4
        if leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, middle_channels, 1, 1, leaky, norm, True)
        self.conv2 = ConvNorm(middle_channels, middle_channels, 3, stride, leaky, norm, True)
        self.conv3 = ConvNorm(middle_channels, out_channels, 1, 1, leaky, norm, False)

        self.identity_mapping = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False)

        self.need_map = in_channels != out_channels or stride != 1

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.need_map:
            identity = self.identity_mapping(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        return out


class ScaleUpsample(nn.Module):
    def __init__(self, use_deconv=False, num_channels=None, scale_factor=None, mode='trilinear', align_corners=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        if use_deconv:
            self.trans_conv = nn.ConvTranspose3d(num_channels, num_channels, kernel_size=3,
                                                stride=scale_factor, padding=1, output_padding=scale_factor - 1)

    def forward(self, x):
        if not self.use_deconv:
            return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        else:
            return self.trans_conv(x)


class AttentionConnection(nn.Module):
    def __init__(self, factor=1.0):
        super().__init__()
        self.param = nn.Parameter(torch.Tensor(1).fill_(factor))

    def forward(self, feature, attention):
        return (self.param + attention) * feature


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, **kwargs):
        super().__init__()
        self.W_g = ConvNorm(F_g, F_int, kernel_size=1, stride=1, activation=False, **kwargs)

        self.W_x = ConvNorm(F_l, F_int, kernel_size=1, stride=2, activation=False, **kwargs)

        self.psi = nn.Sequential(
            ConvNorm(F_int, 1, kernel_size=1, stride=1, activation=False, **kwargs),
            nn.Sigmoid()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * self.upsample(psi)


class ParallelDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2
        self.conv3_0 = ConvNorm(in_channels[0], self.midchannels, 1, 1, **kwargs)
        self.conv4_0 = ConvNorm(in_channels[1], self.midchannels, 1, 1, **kwargs)
        self.conv5_0 = ConvNorm(in_channels[2], self.midchannels, 1, 1, **kwargs)

        self.conv4_5 = ConvNorm(2 * self.midchannels, self.midchannels, 3, **kwargs)
        self.conv3_4 = ConvNorm(2 * self.midchannels, self.midchannels, 3, **kwargs)

        self.conv_out = nn.Conv3d(3 * self.midchannels, out_channels, kernel_size=1)

    def forward(self, x3, x4, x5):
        # x1 has the fewest channels and largest resolution
        # x3 has the most channels and the smallest resolution
        size = x3.shape[2:]

        # first interpolate three feature maps to the same resolution
        f3 = self.conv3_0(x3)  # (None, midchannels, h3, w3)
        f4 = self.conv4_0(F.interpolate(x4, size, mode='trilinear', align_corners=False))  # (None, midchannels, h3, w3)
        level5 = self.conv5_0(F.interpolate(x5, size, mode='trilinear', align_corners=False))  # (None, midchannels, h3, w3)

        # fuse feature maps
        level4 = self.conv4_5(torch.cat([f4, level5], dim=1))  # (None, midchannels, h3, w3)
        level3 = self.conv3_4(torch.cat([f3, level4], dim=1))  # (None, midchannels, h3, w3)

        fused_out_reduced = torch.cat([level3, level4, level5], dim=1)

        out = self.conv_out(fused_out_reduced)

        return out


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2

        self.conv5_4 = ConvNorm(in_channels[2], in_channels[1], 1, 1, **kwargs)
        self.conv4_0 = ConvNorm(in_channels[1], in_channels[1], 3, 1, **kwargs)
        self.conv4_3 = ConvNorm(in_channels[1], in_channels[0], 1, 1, **kwargs)
        self.conv3_0 = ConvNorm(in_channels[0], in_channels[0], 3, 1, **kwargs)

        self.conv_out = nn.Conv3d(in_channels[0], out_channels, kernel_size=1)

    def forward(self, x3, x4, x5):
        # x1 has the fewest channels and largest resolution
        # x3 has the most channels and the smallest resolution
        x5_up = self.conv5_4(F.interpolate(x5, size=x4.shape[2:], mode='trilinear', align_corners=False))
        x4_refine = self.conv4_0(x5_up + x4)
        x4_up = self.conv4_3(F.interpolate(x4_refine, size=x3.shape[2:], mode='trilinear', align_corners=False))
        x3_refine = self.conv3_0(x4_up + x3)

        out = self.conv_out(x3_refine)

        return out

class SENet3D(nn.Module):
    def __init__(self, in_channel, ratio_rate=16):
        super(SENet3D, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
        nn.Linear(in_channel, in_channel // ratio_rate, False),
            nn.ReLU(),
            nn.Linear(in_channel // ratio_rate, in_channel, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w, d = x.size()

        # b, c, h, w, d -> b, c, 1, 1, 1
        avg = self.avg_pool(x).view([b, c])

        # b, c -> b, c // ratio_rate -> b, c -> b, c, 1, 1
        fc = self.fc(avg).view([b, c, 1, 1, 1])

        return x * fc

class SpacialAttention3D(nn. Module):
    def __init__(self, kernel_size=7):
        super(SpacialAttention3D, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, 1, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out,_ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)

        return out
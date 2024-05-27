import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)

class ConvNorm(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = Conv(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(num_groups=2, num_channels=out_channel)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))
        return out

class ConvStem(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = ConvNorm(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.conv2 = ConvNorm(in_channel=out_channel, out_channel=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out

class SMEF(nn.Module):
    def __init__(self, in_channel,  b=1, gama=2):
        super().__init__()
        self.channel_avg = nn.AdaptiveAvgPool3d(1)
        # kernel_size = int(abs((math.log(in_channel, 2)+b)/gama))
        kernel_size = 3
        self.channel_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.spatial_conv = nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1, padding=0)
        self.groupnorm = nn.GroupNorm(num_groups=2, num_channels=in_channel)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # print(x1.shape)
        # print(x2.shape)
        features = x1 + x2
        B, C, Z, Y, X = features.shape
        features_z = F.adaptive_avg_pool3d(features, (Z, 1, 1))
        features_y = F.adaptive_avg_pool3d(features, (1, Y, 1))
        features_x = F.adaptive_avg_pool3d(features, (1, 1, X))

        spatial_features = features_z + features_y + features_x
        # print("spatial", spatial_features.shape)
        spatial_features = self.groupnorm(self.spatial_conv(self.relu(self.groupnorm(self.spatial_conv(spatial_features)))))
        channel_features = self.channel_avg(features)
        channel_features = (self.channel_conv(channel_features.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
                            .unsqueeze(-1).unsqueeze(-1))
        # print(channel_features.shape)
        # print(spatial_features.shape)
        modalities_features = spatial_features+channel_features
        modalities_attn = self.sigmoid(modalities_features)

        return modalities_attn

class SMEF_Module(nn.Module):
    def __init__(self, in_channel, position=True):
        super().__init__()
        self.position = position
        self.msm1 = SMEF(in_channel=in_channel)
        self.msm2 = SMEF(in_channel=in_channel)
        if position:
            self.conv = ConvStem(in_channel=in_channel * 2, out_channel=in_channel * 2)
        else:
            self.conv = ConvStem(in_channel=in_channel, out_channel=in_channel)

    def forward(self, t1, t1ce, t2, flair):
        m1 = self.msm1(t1, t2)
        m2 = 1.0 - m1
        modality1_feature = m1 * t1 + m2 * t2
        m3 = self.msm2(t1ce, flair)
        m4 = 1.0 - m3
        modality2_feature = m3 * t1ce + m4 * flair
        if self.position:
            modality_features = torch.cat([modality1_feature, modality2_feature], dim=1)
            # print(modality_features.shape)
            output = self.conv(modality_features)
        else:
            modality_features = modality1_feature + modality2_feature
            # print(modality_features.shape)
            output = self.conv(modality_features)
        return output
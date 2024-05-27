from module.model_basic_modules import *
from module.model_attention_blocks import FCNHead, ParallelDecoder

from module.SMEF import SMEF_Module
from module.MFCI import MFCI_model

head_list = ['fcn', 'parallel']
head_map = {'fcn': FCNHead,
            'parallel': ParallelDecoder}

class Backbone(nn.Module):
    """
    Model backbone to extract features
    """
    def __init__(self, input_channels=3, channels=(32, 64, 128, 256, 512), strides=(1, 2, 2, 2, 2), **kwargs):
        super().__init__()
        self.nb_filter = channels
        self.strides = strides
        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck

        if kwargs['norm'] == 'GROUP':
            self.conv0_0 = nn.Sequential(
                nn.Conv3d(input_channels, self.nb_filter[0], kernel_size=3, stride=self.strides[0], padding=1),
                nn.ReLU()
            )
        else:
            self.conv0_0 = res_unit(input_channels, self.nb_filter[0], self.strides[0], **kwargs)
        self.conv1_0 = res_unit(self.nb_filter[0], self.nb_filter[1], self.strides[1], **kwargs)
        self.conv2_0 = res_unit(self.nb_filter[1], self.nb_filter[2], self.strides[2], **kwargs)
        self.conv3_0 = res_unit(self.nb_filter[2], self.nb_filter[3], self.strides[3], **kwargs)
        # self.conv4_0 = res_unit(self.nb_filter[3], self.nb_filter[4], self.strides[4], **kwargs)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        return x0_0, x1_0, x2_0, x3_0


class SegmentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        raise NotImplementedError('Forward method must be implemented before calling it!')

    def predictor(self, x):
        return self.forward(x)['out']

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        loss_out = criterion(outputs['out'], targets)
        if is_ds:
            loss_3 = criterion(outputs['level3'], targets)
            loss_2 = criterion(outputs['level2'], targets)
            loss_1 = criterion(outputs['level1'], targets)
            multi_loss = loss_out + loss_3 + loss_2 + loss_1
        else:
            multi_loss = loss_out
        return multi_loss

class CFCI(SegmentationNetwork):
    def __init__(self, num_classes, input_channels=1, channels=(16, 32, 64, 128), use_deconv=False,strides=(1, 2, 2, 2),
                 mfc_embed_size=512, patch_size=(16, 16, 16), mode_layers=8, mode_embed_size=128,
                 mode_attention_heads=12, mfi_layers=8, mfi_embed_size=256, mfi_attention_heads=12, mfi_hidden_size=512,
                 dropout=0.1, **kwargs):
        super().__init__()
        self.t1_backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        self.t2_backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        self.t1ce_backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)
        self.flair_backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides, **kwargs)

        self.mfci = MFCI_model(mfc_embed_size=mfc_embed_size, patch_size=patch_size, mode_layers=mode_layers, mode_embed_size=mode_embed_size,
                               mode_attention_heads=mode_attention_heads, mfi_layers=mfi_layers, mfi_embed_size=mfi_embed_size,
                               mfi_attention_heads=mfi_attention_heads, mfi_hidden_size=mfi_hidden_size, dropout=dropout)

        self.enc1_smef = SMEF_Module(in_channel=channels[1])
        self.enc2_smef = SMEF_Module(in_channel=channels[2])
        self.enc3_smef = SMEF_Module(in_channel=channels[3])

        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck
        self.conv_bottom = res_unit(channels[3] * 2, channels[3] * 2, lkdw=True, **kwargs)
        self.conv3_1 = res_unit(channels[3] * 4, channels[3] * 2, lkdw=True, **kwargs)
        self.conv2_1 = res_unit(channels[2] * 4, channels[2] * 2, lkdw=True, **kwargs)
        self.conv1_1 = res_unit(channels[1] * 4, channels[1] * 2, lkdw=True, **kwargs)
        self.conv0_1 = res_unit(channels[0] * 2, channels[0], lkdw=True, **kwargs)

        self.up3 = nn.Sequential(
            ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-1], scale_factor=strides[-1]),
            res_unit(channels[3] * 2, channels[3], lkdw=True, **kwargs)
        )
        self.up2 = nn.Sequential(
            ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-2], scale_factor=strides[-2]),
            res_unit(channels[2] * 2, channels[2], lkdw=True, **kwargs)
        )
        self.up1 = nn.Sequential(
            ScaleUpsample(use_deconv=use_deconv, num_channels=channels[-3], scale_factor=strides[-3]),
            res_unit(channels[1] * 2, channels[1], lkdw=True, **kwargs)
        )

        # deep supervision
        self.convds3 = nn.Conv3d(channels[3] * 2, num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(channels[2] * 2, num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(channels[1] * 2, num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        t1, t2, t1ce, flair = x[:, 0:1, :, :, :], x[:, 1:2, :, :, :], x[:, 2:3, :, :, :], x[:, 3:4, :, :, :]
        t1_x0, t1_x1, t1_x2, t1_x3 = self.t1_backbone(t1)
        t2_x0, t2_x1, t2_x2, t2_x3 = self.t2_backbone(t2)
        t1ce_x0, t1ce_x1, t1ce_x2, t1ce_x3 = self.t1ce_backbone(t1ce)
        flair_x0, flair_x1, flair_x2, flair_x3 = self.flair_backbone(flair)

        bottom_feature = self.mfci(t1_x3, t2_x3, t1ce_x3, flair_x3)
        bottom_feature = self.conv_bottom(bottom_feature)

        smef_de3 = self.enc3_smef(t1_x3, t1ce_x3, t2_x3, flair_x3)
        smef_de2 = self.enc2_smef(t1_x2, t1ce_x2, t2_x2, flair_x2)
        smef_de1 = self.enc1_smef(t1_x1, t1ce_x1, t2_x1, flair_x1)

        # Decoding
        x3_d = self.conv3_1(torch.cat([bottom_feature, smef_de3], dim=1))
        x2_d = self.conv2_1(torch.cat([self.up3(x3_d), smef_de2], dim=1))
        x1_d = self.conv1_1(torch.cat([self.up2(x2_d), smef_de1], dim=1))
        x0_d = self.conv0_1(self.up1(x1_d))

        out = dict()
        out['out'] = [F.interpolate(self.convds3(x3_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds2(x2_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds1(x1_d), size=size, mode='trilinear', align_corners=False),
                      F.interpolate(self.convds0(x0_d), size=size, mode='trilinear', align_corners=False)]
        return out

    def predictor(self, x):
        return self.forward(x)['out'][-1]

    def get_multi_loss(self, criterion, outputs, targets, is_ds=False):
        if is_ds:
            ## 固定权重
            loss_weight = [0.2, 0.3, 0.6, 0.8]  # [0.8, 0.6, 0.3, 0.2]
            loss_out_3 = loss_weight[3] * criterion(outputs['out'][3], targets)
            loss_out_2 = loss_weight[2] * criterion(outputs['out'][2], targets)
            loss_out_1 = loss_weight[1] * criterion(outputs['out'][1], targets)
            loss_out_0 = loss_weight[0] * criterion(outputs['out'][0], targets)
            loss_out = loss_out_3 + loss_out_2 + loss_out_1 + loss_out_0
        else:
            loss_out = criterion(outputs['out'][-1], targets)

        multi_loss = loss_out
        return multi_loss



# @DC
if __name__ == "__main__":
    model = CFCI(num_classes=4, input_channels=1, channels=(12, 24, 48, 96), use_deconv=False, strides=(1, 2, 2, 2),
                 mfc_embed_size=384, patch_size=(16, 16, 16), mode_layers=4, mode_embed_size=96,
                 mode_attention_heads=8, mfi_layers=4, mfi_embed_size=192, mfi_attention_heads=8, mfi_hidden_size=384,
                 dropout=0.1, leaky=True, norm='INSTANCE')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'There are {n_params} trainable parameters.')

    batch_size = 1
    x = torch.randn([batch_size, 4, 128, 128, 128])
    from thop import profile

    flops, params = profile(model, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
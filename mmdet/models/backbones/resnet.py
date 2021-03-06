import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from ..dcn_v2.modules import ConvOffset2d
from ..mod_dcn.dcn_v2 import DCNv2
from ..cc_attention.functions import CrissCrossAttention

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 with_dcn=False,
                 num_deformable_groups=1,
                 dcn_offset_lr_mult=0.1,
                 use_regular_conv_on_stride=False,
                 use_modulated_dcn=False,
                 use_non_local=False,
                 non_local_position='after_relu',
                 non_local_recurrence=2):
        """Bottleneck block.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        if style == 'pytorch':
            conv1_stride = 1
            conv2_stride = stride
        else:
            conv1_stride = stride
            conv2_stride = 1
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=conv1_stride, bias=False)

        self.with_dcn = with_dcn
        self.use_modulated_dcn = use_modulated_dcn
        if use_regular_conv_on_stride and stride > 1:
            self.with_dcn = False
        if self.with_dcn:
            print("--->> use {}dcn in block where c_in={} and c_out={}".format(
                'modulated ' if self.use_modulated_dcn else '', inplanes, planes))
            if use_modulated_dcn:
                self.conv_offset_mask = nn.Conv2d(
                    planes,
                    num_deformable_groups * 27,
                    kernel_size=3,
                    stride=conv2_stride,
                    padding=dilation,
                    dilation=dilation)
                self.conv_offset_mask.lr_mult = dcn_offset_lr_mult
                self.conv_offset_mask.zero_init = True

                self.conv2 = DCNv2(planes, planes, 3, stride=conv2_stride,
                                          padding=dilation, dilation=dilation,
                                          deformable_groups=num_deformable_groups, no_bias=True)
            else:
                self.conv2_offset = nn.Conv2d(
                    planes,
                    num_deformable_groups * 18,
                    kernel_size=3,
                    stride=conv2_stride,
                    padding=dilation,
                    dilation=dilation)
                self.conv2_offset.lr_mult = dcn_offset_lr_mult
                self.conv2_offset.zero_init = True

                self.conv2 = ConvOffset2d(planes, planes, (3, 3), stride=conv2_stride,
                    padding=dilation, dilation=dilation,
                    num_deformable_groups=num_deformable_groups)
        else:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)


        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.use_non_local = use_non_local
        if self.use_non_local:
            self.non_local_position = non_local_position
            self.non_local_recurrence = non_local_recurrence
            if self.non_local_position in ['before_conv2', 'after_conv2']:
                self.non_local_block = CrissCrossAttention(planes)
            elif self.non_local_position == 'after_relu':
                self.non_local_block = CrissCrossAttention(planes * self.expansion)
            else:
                assert False, 'non local position {} not supported!'.format(self.non_local_position)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_recurrence_non_local(x):
            for _ in range(self.non_local_recurrence):
                x = self.non_local_block(x)
            return x

        def _inner_forward(x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            if self.use_non_local and self.non_local_position == 'before_conv2':
                print("--->> non local ran before conv2")
                out = _inner_recurrence_non_local(out)

            if self.with_dcn:
                if self.use_modulated_dcn:
                    offset_mask = self.conv_offset_mask(out)
                    offset1, offset2, mask_raw = torch.chunk(offset_mask, 3, dim=1)
                    offset = torch.cat((offset1, offset2), dim=1)
                    mask = torch.sigmoid(mask_raw)
                    out = self.conv2(out, offset, mask)
                else:
                    offset = self.conv2_offset(out)
                    out = self.conv2(out, offset)
            else:
                out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            if self.use_non_local and self.non_local_position == 'after_conv2':
                print("--->> non local ran after conv2")
                out = _inner_recurrence_non_local(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            return out


        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        if self.use_non_local and self.non_local_position == 'after_relu':
            out = _inner_recurrence_non_local(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   with_dcn=False,
                   dcn_offset_lr_mult=0.1,
                   use_regular_conv_on_stride=False,
                   use_modulated_dcn=False,
                   non_local_position='after_relu',
                   non_local_recurrence=2,
                   non_local_res_blocks=[]):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp,
            with_dcn=with_dcn,
            dcn_offset_lr_mult=dcn_offset_lr_mult,
            use_regular_conv_on_stride=use_regular_conv_on_stride,
            use_modulated_dcn=use_modulated_dcn,
            use_non_local=(0 in non_local_res_blocks),
            non_local_position=non_local_position,
            non_local_recurrence=non_local_recurrence))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation, style=style, with_cp=with_cp, with_dcn=with_dcn, 
                  dcn_offset_lr_mult=dcn_offset_lr_mult, use_regular_conv_on_stride=use_regular_conv_on_stride,
                  use_modulated_dcn=use_modulated_dcn, use_non_local=(i in non_local_res_blocks),
                  non_local_position=non_local_position, non_local_recurrence=non_local_recurrence))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 with_cp=False,
                 with_dcn=False,
                 dcn_start_stage=3,
                 dcn_offset_lr_mult=0.1,
                 use_regular_conv_on_stride=False,
                 use_modulated_dcn=False,
                 non_local_flag_all_stages=[[], [], [], []],
                 non_local_position='after_relu',
                 non_local_recurrence=2):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        assert num_stages >= 1 and num_stages <= 4
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) < num_stages

        self.out_indices = out_indices
        self.style = style
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.with_cp = with_cp

        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            use_dcn = False
            if with_dcn and i >= (dcn_start_stage - 2):
                use_dcn = True
            res_layer = make_res_layer(
                block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                with_dcn=use_dcn,
                dcn_offset_lr_mult=dcn_offset_lr_mult,
                use_regular_conv_on_stride=use_regular_conv_on_stride,
                use_modulated_dcn=use_modulated_dcn,
                non_local_res_blocks=non_local_flag_all_stages[i],
                non_local_position=non_local_position,
                non_local_recurrence=non_local_recurrence)
            self.inplanes = planes * block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = block.expansion * 64 * 2**(len(stage_blocks) - 1)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
            for m in self.modules():
                if isinstance(m, nn.Conv2d) and hasattr(m, 'zero_init') and m.zero_init:
                    constant_init(m, 0)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if hasattr(m, 'zero_init') and m.zero_init:
                        constant_init(m, 0)
                    else:
                        kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False

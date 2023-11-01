from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DownsampledMultiheadAttention2D(nn.Module):
    """
        Downsampled multi-headed attention based on
        A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, Ll. Jones, A.N. Gomez,
        L. Kaiser, I. Polosukhin "Attention Is All You Need"
        https://arxiv.org/abs/1706.03762
        """

    def __init__(
            self, in_features, att_features, blocks=8, downsampling=2,
            norm=partial(torch.softmax, dim=1),
    ):
        super().__init__()
        assert att_features % blocks == 0,\
            'The number of attention features must be divisible by ' \
            'the number of blocks'
        self.blocks = blocks
        self.out_features = att_features
        self.features = att_features // blocks
        self.downsampling = downsampling
        self.sa_blocks = nn.ModuleList([
            DownsampledSelfAttention2D(
                in_features, self.features, downsampling, norm
            )
            for _ in range(self.blocks)
        ])
        self.final_block = nn.Conv2d(in_features * blocks, in_features, 1)

    def forward(self, x):
        x = torch.cat([sa_i(x) for sa_i in self.sa_blocks], dim=1)
        z = self.final_block(x)
        return z


class DownsampledSelfAttention2D(nn.Module):
    """
    Downsampled non-local self-attention block based on
    X. Wang, R. Girshick, A.Gupta, K. He "Non-local Neural Networks"
    https://arxiv.org/abs/1711.07971
    """

    def __init__(
            self, in_features, att_features, downsampling=2,
            norm=partial(torch.softmax, dim=1),
    ):
        super().__init__()
        self.features = att_features
        self.downsampling = downsampling
        self.conv_theta = nn.Sequential(
            nn.Conv2d(
                in_features, att_features, downsampling,
                stride=downsampling
            ),
            nn.InstanceNorm2d(att_features)
        )
        self.conv_phi = nn.Sequential(
            nn.Conv2d(
                in_features, att_features, downsampling,
                stride=downsampling
            ),
            nn.InstanceNorm2d(att_features)
        )
        self.conv_g = nn.Conv2d(
            in_features, att_features, downsampling, stride=downsampling
        )
        self.conv_final = nn.Conv3d(att_features, in_features, 1)
        self.norm = norm

    def forward(self, x):
        theta = self.conv_theta(x).flatten(2).transpose(1, 2)
        phi = self.conv_phi(x).flatten(2)
        g = self.conv_g(x).flatten(2)
        ds_x = F.max_pool3d(x, self.downsampling)

        att = torch.bmm(theta, phi)
        att_map = self.norm(
            att.flatten(1) / np.sqrt(self.features)
        ).view_as(att)
        ds_self_att = self.conv_final(
            torch.bmm(g, att_map).view(
                (ds_x.shape[0], g.shape[1]) + ds_x.shape[2:]
            )
        )

        return ds_self_att


class AttentionGate2D(nn.Module):
    """
    Attention gate block based on
    Jo Schlemper, Ozann Oktay, Michiel Schaap, Mattias Heinrich, Bernhard
    Kainz, Ben Glocker, Daniel Rueckert. "Attention gated networks: Learning
    to leverage salient regions in medical images"
    https://doi.org/10.1016/j.media.2019.01.012
    """

    def __init__(
            self, x_features, g_features, int_features, regions=1,
            sigma2=torch.sigmoid
    ):
        super().__init__()
        self.conv_g = nn.Conv2d(g_features, int_features, 1)
        self.conv_x = nn.Conv2d(x_features, int_features, 1)
        self.conv_phi = nn.Conv2d(int_features, regions, 1)
        self.sigma2 = sigma2
        self.regions = regions

    def forward(self, x, g, attention=False):
        x_emb = self.conv_x(x)
        g_emb = self.conv_g(
            F.interpolate(
                g, size=x_emb.size()[2:], mode='trilinear',
                align_corners=False
            )
        )
        phi_emb = self.conv_phi(F.relu(g_emb + x_emb))
        alpha = self.sigma2(phi_emb)

        if self.regions > 1:
            x = torch.cat(
                [x * alpha_i for alpha_i in torch.split(alpha, 1, dim=1)],
                dim=1
            )
        else:
            x = x * alpha

        if attention:
            output = x, alpha
        else:
            output = x

        return output
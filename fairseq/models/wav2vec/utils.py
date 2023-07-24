# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

from .cif import cif_function


def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder


class Subsampler(nn.Module):
    def __init__(self, ratio=4, hidden_dim=768):
        super().__init__()
        self.layer = nn.AvgPool1d(kernel_size=ratio, stride=ratio)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layer(x)
        x = x.transpose(1, 2)
        return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class AlphaModule(nn.Module):
    def __init__(self, layer_type, embed_dim, conv_width):
        super().__init__()

        self.layer_type = layer_type

        if layer_type == 'cnn':
            self.cnn = nn.Conv1d(embed_dim, embed_dim, conv_width, stride=1, padding=int(conv_width / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

            self.layer = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1)
            )

    def forward(self, x):
        if self.layer_type == 'cnn':
            x = x.transpose(1, 2)
            x = self.cnn(x)
            x = x.transpose(1, 2)

        x = self.layer(x)
        x = torch.sigmoid(x).squeeze(dim=-1) # weight has shape B x T x 1

        return x


class CIF(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        beta=1.0,
        layer_type="cnn",
        conv_width=5,
        mixup=False,
        mixup_a=1.0,
        mixup_distro="uniform",
        upper_lambd=2.0,
    ):
        super().__init__()

        self.alpha_fn = AlphaModule(layer_type, embed_dim, conv_width)
        self.beta = beta
        self.mixup = mixup
        self.mixup_a = mixup_a  # parameter for beta distribution
        self.mixup_distro = mixup_distro
        self.upper_lambd = upper_lambd
        if self.upper_lambd == 2.0:
            self.upper_lambd -= 1e-5 # to prevent zero alphas

        print(f"mixup_distro: {mixup_distro}, upper_lambd: {self.upper_lambd}")

    def forward(
        self,
        x,
        pad_mask=None,
        target_lengths=None,
        gamma=0.0,
        is_training=True,
        overwrite_alpha=None,
        overwrite_lambd=None,
    ):
        """
        (1 - gamma) * target_len + gamma * predicted_alpha.
        """
        if overwrite_alpha is None:
            alpha = self.alpha_fn(x)
            if self.mixup:
                if overwrite_lambd is None:
                    if self.mixup_distro == "uniform":
                        lambd = np.random.beta(self.mixup_a, self.mixup_a)
                    elif self.mixup_distro == "uniform-fr":
                        fr = np.random.beta(1.0, 1.0) * 70 + 20 # uniform from 20ms to 90 ms
                        lambd = 9 / 7 * (1 - 20 / fr)
                    elif self.mixup_distro == "scale":
                        lambd = np.random.beta(self.mixup_a, self.mixup_a) + 1 # from 1 to 2
                    elif self.mixup_distro == "both":
                        lambd = np.random.beta(1.0, 1.0) * self.upper_lambd
                    else:
                        raise NotImplementedError
                else:
                    lambd = overwrite_lambd

                ones = torch.ones_like(alpha)

                if type(lambd) is not torch.Tensor:
                    # this will break gradient
                    lambd = torch.tensor(lambd).to(alpha.device)

                mixed_alpha = lambd * alpha + (1.0 - lambd) * ones

                weight_1 = F.relu(1 - lambd)
                weight_2 = F.relu(lambd) - 2 * F.relu(lambd - 1)
                mixed_alpha = weight_1 * ones + weight_2 * alpha

            else:
                mixed_alpha = None
        else:
            lambd = None
            mixed_alpha = None
            alpha = overwrite_alpha.to(x.device)

        padding_mask = torch.logical_not(pad_mask) if pad_mask is not None else None
        out = cif_function(
            x,
            alpha if mixed_alpha is None else mixed_alpha,
            beta=self.beta,
            gamma=gamma,
            padding_mask=padding_mask,
            target_lengths=target_lengths,
            is_training=is_training,
        )

        # print(f"lambda = {lambd}, length = {out['cif_lengths'][0].float().mean()} / {x.size(1)} = {out['cif_lengths'][0].float().mean() / x.size(1)}")

        return out, alpha



# class CIF(nn.Module):
#     def __init__(self, embed_dim=512, beta=1.0, layer_type='cnn', conv_width=5):
#         super().__init__()

#         self.alpha_fn = AlphaModule(layer_type, embed_dim, conv_width)
#         self.beta = beta

#     def forward(self, x, pad_mask=None, target_lengths=None, gamma=0.0, is_training=True):
#         """
#             (1 - gamma) * target_len + gamma * predicted_alpha.
#         """
#         alpha = self.alpha_fn(x)
#         padding_mask = torch.logical_not(pad_mask) if pad_mask is not None else None
#         out = cif_function(
#             x,
#             alpha,
#             beta=self.beta,
#             gamma=gamma,
#             padding_mask=padding_mask,
#             target_lengths=target_lengths,
#             is_training=is_training
#         )

#         return out

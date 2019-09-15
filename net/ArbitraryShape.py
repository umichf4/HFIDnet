# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-09-04 14:27:09
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-14 20:05:52

import os
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class GeneratorNet(nn.Module):
    def __init__(self, noise_dim=100, ctrast_dim=58, d=32):
        super().__init__()
        self.noise_dim = noise_dim
        self.ctrast_dim = ctrast_dim
        self.deconv_block_ctrast = nn.Sequential(
            nn.ConvTranspose2d(self.ctrast_dim, d * 4, 4, 1, 0),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
        )
        self.deconv_block_noise = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, d * 4, 4, 1, 0),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
        )
        self.deconv_block_cat = nn.Sequential(
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d * 2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.ConvTranspose2d(d, 1, 4, 2, 1),
            nn.Tanh()
        )
        self.fc_block_1 = nn.Sequential(
            nn.Linear(64 * 64, 64 * 16),
            nn.BatchNorm1d(64 * 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(64 * 16, 64 * 4),
            nn.BatchNorm1d(64 * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(64 * 4, ctrast_dim),
            nn.BatchNorm1d(ctrast_dim),
            nn.LeakyReLU(0.2)
        )
        self.fc_block_2 = nn.Sequential(
            nn.Linear(ctrast_dim, 2),
            nn.Tanh()
        )
        self.short_cut = nn.Sequential(
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, noise_in, ctrast_in):
        noise = self.deconv_block_noise(noise_in.view(-1, self.noise_dim, 1, 1))
        ctrast = self.deconv_block_ctrast(ctrast_in.view(-1, self.ctrast_dim, 1, 1))
        net = torch.cat((noise, ctrast), 1)
        img = self.deconv_block_cat(net)
        pair_in = self.fc_block_1(img.view(img.size(0), -1)) + self.short_cut(ctrast_in.view(ctrast_in.size(0), -1))
        pair = self.fc_block_2(pair_in)
        return (img + 1) / 2, (pair + 1) / 2


class SimulatorNet(nn.Module):
    def __init__(self, spec_dim=58, d=32):
        super().__init__()
        self.spec_dim = spec_dim
        self.conv_block_shape = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(1, 3 * d // 4, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_gatk = nn.Sequential(
            # ------------------------------------------------------
            nn.ReplicationPad2d((63, 0, 63, 0)),
            nn.Conv2d(1, d // 8, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.conv_block_cat = nn.Sequential(
            # ------------------------------------------------------
            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 4, d * 8, 4, 2, 1),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # ------------------------------------------------------
            nn.Conv2d(d * 8, d * 16, 4, 1, 0),
            nn.Sigmoid()
        )
        self.fc_block = nn.Sequential(
            nn.Linear(d * 16, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # nn.Linear(512, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.2),
            nn.Linear(128, spec_dim),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, shape_in, gap_in, thk_in):
        shape = self.conv_block_shape(shape_in)
        gap = self.conv_block_gatk(gap_in.view(-1, 1, 1, 1))
        thk = self.conv_block_gatk(thk_in.view(-1, 1, 1, 1))
        net = torch.cat((shape, gap, thk), 1)
        spec = self.conv_block_cat(net)
        spec = self.fc_block(spec.view(spec.shape[0], -1))
        return (spec + 1) / 2


if __name__ == '__main__':
    import torchsummary

    # if torch.cuda.is_available():
    #     generator = GeneratorNet().cuda()
    # else:
    #     generator = GeneratorNet()

    # torchsummary.summary(generator, [tuple([100]), tuple([58])])

    if torch.cuda.is_available():
        simulator = SimulatorNet().cuda()
    else:
        simulator = SimulatorNet()

    torchsummary.summary(simulator, [(1, 64, 64), tuple([1]), tuple([1])])

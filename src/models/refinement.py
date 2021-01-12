import torch
import torch.nn as nn
import numpy as np


class Block(nn.Module):
    """Simple Block for the vitruvian model"""
    def __init__(self, in_planes: int, out_planes: int, drop, BN, activation):
        super(Block, self).__init__()
        features = [nn.Linear(in_planes, out_planes)]
        if BN:
            features += [nn.BatchNorm1d(out_planes)]
        if activation is not None:
            if activation.__name__ == 'ReLU':
                features += [activation(inplace=True)]
            else:
                features += [activation()]
        if drop:
            features += [nn.Dropout(0.2)]
        self.features = nn.Sequential(*features)

    def forward(self, x):
        x = self.features(x)
        return x


class LinearModel(nn.Module):
    """Linear Model for the vitruvian block"""
    def __init__(self, in_planes: int, out_planes: int, linear_size, drop: bool = False,
                 bn: bool = False, residual: bool = True, activation=nn.ReLU,
                 attention: bool = False, bigskip: bool = False):
        super(LinearModel, self).__init__()
        self.residual = residual
        self.attention = attention
        self.bigskip = bigskip
        self.inputBlock = nn.Sequential(*[Block(in_planes, linear_size, drop, bn, activation)])
        self.Block1 = nn.Sequential(*[Block(linear_size, linear_size, drop, bn, activation),
                                      Block(linear_size, linear_size, drop, bn, activation)])
        self.Block2 = nn.Sequential(*[Block(linear_size, linear_size, drop, bn, activation),
                                      Block(linear_size, linear_size, drop, bn, activation)])
        self.outputBlock = nn.Sequential(*[Block(linear_size, out_planes, False, False, None)])

        if self.attention:
            self.attentionBlock = nn.Sequential(*[Block(linear_size, out_planes, False, False, nn.Sigmoid)])

        if self.bigskip:
            self.skipBlock = nn.Sequential(*[Block(out_planes * 2, out_planes * 2, drop, bn, activation),
                                             Block(out_planes * 2, out_planes, False, False, None)])

    def forward(self, x):
        size = x.size()[1:]
        n_features = np.prod(size)
        x = x.view(-1, n_features)

        inp = x

        x = self.inputBlock(x)

        if self.attention:
            attention = self.attentionBlock(x)

        residual = x
        x = self.Block1(x)
        x = x + residual if self.residual else x

        residual = x
        x = self.Block2(x)
        x = x + residual if self.residual else x

        x = self.outputBlock(x)

        if self.bigskip:
            x = self.skipBlock(torch.cat((x, inp), dim=1))

        if self.attention:
            x = attention * x

        x = x.view(-1, size[0], size[1])
        return x
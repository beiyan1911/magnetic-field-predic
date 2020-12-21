import torch.nn as nn


def NormLayer(num_groups, num_channels):
    return nn.GroupNorm(num_groups, num_channels)

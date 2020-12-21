import torch
import torch.nn as nn
from model.norm_layer import NormLayer


class MIMN(nn.Module):
    def __init__(self, filter_size, num_hidden, norm):
        super(MIMN, self).__init__()

        self.filter_size = filter_size
        self.pad = filter_size // 2
        self.num_hidden = num_hidden
        self.norm = norm
        self._forget_bias = 1.0

        # h_t
        self.h_t = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=self.pad)
        # c_t
        self.c_t = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=self.pad)
        # x
        self.x = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=self.pad)

        # bn
        self.bn_h_concat = NormLayer(4, self.num_hidden * 4)
        self.bn_x_concat = NormLayer(4, self.num_hidden * 4)

    def forward(self, x, h_t, c_t):
        # h
        h_concat = self.h_t(h_t)
        if self.norm:
            h_concat = self.bn_h_concat(h_concat)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, 1)

        # x
        x_concat = self.x(x)
        if self.norm:
            x_concat = self.bn_x_concat(x_concat)
        i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, 1)

        # c
        # ct_activation = torch.mul(c_t.repeat([1, 3, 1, 1]), self.ct_weight)
        ct_activation = self.c_t(c_t)
        i_c, f_c, g_c, o_c = torch.split(ct_activation, self.num_hidden, 1)

        i_ = i_h + i_c + i_x
        f_ = f_h + f_c + f_x
        g_ = g_h + g_c + g_x
        o_ = o_h + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new  # [batch, in_height, in_width, num_hidden]

import torch
import torch.nn as nn

from model.norm_layer import NormLayer


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, filter_size, num_hidden_in, num_hidden, x_shape_in, norm):
        super(SpatioTemporalLSTMCell, self).__init__()
        self.filter_size = filter_size
        self.pad = filter_size // 2
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden
        self.x_shape_in = x_shape_in
        self.layer_norm = norm
        self._forget_bias = 1.0

        # h
        self.h_cc = nn.Conv2d(self.num_hidden_in, self.num_hidden * 4, self.filter_size, 1, padding=self.pad)

        # m
        self.m_cc = nn.Conv2d(self.num_hidden_in, self.num_hidden * 4, self.filter_size, 1, padding=self.pad)
        # x
        self.x_cc = nn.Conv2d(self.x_shape_in, self.num_hidden * 4, self.filter_size, 1, padding=self.pad)
        # c
        self.c_cc = nn.Conv2d(self.num_hidden * 2, self.num_hidden, 1, 1, padding=0)

        # bn
        self.bn_h_cc = NormLayer(4, self.num_hidden * 4)
        self.bn_m_cc = NormLayer(4, self.num_hidden * 4)
        self.bn_x_cc = NormLayer(4, self.num_hidden * 4)

    def forward(self, x, h, c, m):
        # x [batch, in_channels, in_height, in_width]
        # h c m [batch, num_hidden, in_height, in_width]

        h_cc = self.h_cc(h)
        m_cc = self.m_cc(m)
        x_cc = self.x_cc(x)

        if self.layer_norm:
            h_cc = self.bn_h_cc(h_cc)
            m_cc = self.bn_m_cc(m_cc)
            x_cc = self.bn_x_cc(x_cc)

        i_m, g_m, f_m, o_m = torch.split(m_cc, self.num_hidden, 1)  # [batch, num_hidden, in_height, in_width]
        i_h, g_h, f_h, o_h = torch.split(h_cc, self.num_hidden, 1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, 1)

        i = torch.sigmoid(i_x + i_h)
        g = torch.tanh(g_x + g_h)
        f = torch.sigmoid(f_x + f_h + self._forget_bias)
        new_c = f * c + i * g

        i_ = torch.sigmoid(i_x + i_m)
        g_ = torch.tanh(g_x + g_m)
        f_ = torch.sigmoid(f_x + f_m + self._forget_bias)
        new_m = f_ * m + i_ * g_

        o = torch.sigmoid(o_x + o_h + o_m)
        cell = torch.cat((new_c, new_m), 1)
        cell = self.c_cc(cell)
        new_h = o * torch.tanh(cell)

        # [batch, num_hidden, in_height, in_width]
        return new_h, new_c, new_m

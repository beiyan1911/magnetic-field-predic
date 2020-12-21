import torch
import torch.nn as nn

from model.norm_layer import NormLayer


class MIMBlock(nn.Module):
    def __init__(self, filter_size, num_hidden_in, num_hidden, x_shape_in, norm):
        super(MIMBlock, self).__init__()

        self.filter_size = filter_size
        self.pad = filter_size // 2
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden
        self.x_shape_in = x_shape_in
        self.layer_norm = norm
        self._forget_bias = 1.0

        # *********** MIMS
        # h_t
        self.mims_h_t = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=self.pad)
        # c_t
        self.c_t = nn.Conv2d(self.num_hidden, self.num_hidden * 3, self.filter_size, 1, padding=self.pad)
        # x
        self.mims_x = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=self.pad)
        # oc
        # bn
        self.bn_h_concat = NormLayer(4, self.num_hidden * 4)
        self.bn_x_concat = NormLayer(4, self.num_hidden * 4)

        # *********** MIMBLOCK
        # h
        self.t_cc = nn.Conv2d(self.num_hidden_in, self.num_hidden * 3, self.filter_size, 1, padding=self.pad)
        # m
        self.s_cc = nn.Conv2d(self.num_hidden_in, self.num_hidden * 4, self.filter_size, 1, padding=self.pad)
        # x
        self.x_cc = nn.Conv2d(self.x_shape_in, self.num_hidden * 4, self.filter_size, 1, padding=self.pad)
        # c
        self.c_cc = nn.Conv2d(self.num_hidden * 2, self.num_hidden, 1, 1, padding=0)
        # bn
        self.bn_t_cc = NormLayer(3, self.num_hidden * 3)
        self.bn_s_cc = NormLayer(4, self.num_hidden * 4)
        self.bn_x_cc = NormLayer(4, self.num_hidden * 4)

    def MIMS(self, x, h_t, c_t):
        # h_t   c_t[batch, in_height, in_width, num_hidden]
        h_concat = self.mims_h_t(h_t)
        if self.layer_norm:
            h_concat = self.bn_h_concat(h_concat)

        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, 1)

        # x
        x_concat = self.mims_x(x)
        if self.layer_norm:
            x_concat = self.bn_x_concat(x_concat)
        i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, 1)

        ct_activation = self.c_t(c_t)
        i_c, f_c, o_c = torch.split(ct_activation, self.num_hidden, 1)

        i_ = i_h + i_c + i_x
        f_ = f_h + f_c + f_x
        g_ = g_h + g_x
        o_ = o_h + o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new

    def forward(self, x, diff_h, h, c, m, convlstm_c):
        # h
        t_cc = self.t_cc(h)
        # m
        s_cc = self.s_cc(m)
        # x
        x_cc = self.x_cc(x)

        if self.layer_norm:
            t_cc = self.bn_t_cc(t_cc)
            s_cc = self.bn_s_cc(s_cc)
            x_cc = self.bn_x_cc(x_cc)

        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, 1)
        i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, 1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, 1)

        # ************ update M
        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_

        # ************ update C
        c, convlstm_c = self.MIMS(diff_h, c, convlstm_c)
        new_c = c + i * g
        cell = torch.cat((new_c, new_m), 1)
        # c
        cell = self.c_cc(cell)
        new_h = o * torch.tanh(cell)
        # [batch, in_height, in_width, num_hidden]
        return new_h, new_c, new_m, convlstm_c

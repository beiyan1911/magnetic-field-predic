import torch
import torch.nn as nn

from model.st_lstm_cell import SpatioTemporalLSTMCell as stlstm
from model.mim_block import MIMBlock as mimblock
from model.mimn import MIMN as mimn


class MIM(nn.Module):
    def __init__(self, num_layers, num_hidden, config, norm=True):
        super(MIM, self).__init__()
        self.uniform_size = config.uniform_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.filter_size = config.filter_size
        self.total_length = config.input_length + config.predict_length
        self.input_length = config.input_length
        self.norm = norm
        self.width = config.img_width
        self.height = config.img_height
        self.img_channel = config.img_channel
        self.device = config.device

        self.stlstm_layer = nn.ModuleList()  # store stlstm and mimblock
        self.stlstm_layer_diff = nn.ModuleList()  # store mimn

        for i in range(self.num_layers):
            if i == 0:
                num_hidden_in = self.num_hidden[self.num_layers - 1]
            else:
                num_hidden_in = self.num_hidden[i - 1]

            if i < 1:  # first layer use stlstm
                new_stlstm_layer = stlstm(self.filter_size,
                                          num_hidden_in,
                                          self.num_hidden[i],
                                          self.img_channel,
                                          norm=self.norm)
            else:  # other layers use mimblock
                new_stlstm_layer = mimblock(self.filter_size,
                                            num_hidden_in,
                                            self.num_hidden[i],
                                            self.num_hidden[i - 1],
                                            norm=self.norm)
            self.stlstm_layer.append(new_stlstm_layer)

        for i in range(self.num_layers - 1):  # add MIMN
            new_stlstm_layer = mimn(self.filter_size,
                                    self.num_hidden[i + 1],
                                    norm=self.norm)
            self.stlstm_layer_diff.append(new_stlstm_layer)

        # generate image
        self.x_gen = nn.Conv2d(self.num_hidden[self.num_layers - 1], self.img_channel, 1, 1, padding=0)

    def forward(self, images, schedual_sampling_bool):
        if not self.uniform_size:
            batch, _, __, self.height, self.width = images.shape
        else:
            batch = images.shape[0]
        cell_state = []
        hidden_state = []

        cell_state_diff = []
        hidden_state_diff = []
        mims_cell = []
        st_memory = torch.zeros((batch, self.num_hidden[0], self.height, self.width), dtype=torch.float32).to(
            self.device)

        for i in range(self.num_layers):
            zeros = torch.zeros((batch, self.num_hidden[i], self.height, self.width), dtype=torch.float32).to(
                self.device)
            cell_state.append(zeros)
            hidden_state.append(zeros)

        for i in range(self.num_layers - 1):  # add MIMN
            zeros = torch.zeros((batch, self.num_hidden[i], self.height, self.width), dtype=torch.float32).to(
                self.device)
            cell_state_diff.append(zeros)
            hidden_state_diff.append(zeros)
            mims_cell.append(zeros)

        gen_images = []  # store images
        for time_step in range(self.total_length - 1):
            if time_step < self.input_length:
                x_gen = images[:, time_step]  # [batch, in_channel, in_height, in_width]
            else:
                #  mask
                x_gen = schedual_sampling_bool[:, time_step - self.input_length] * images[:, time_step] + \
                        (1 - schedual_sampling_bool[:, time_step - self.input_length]) * x_gen

            preh = hidden_state[0]
            hidden_state[0], cell_state[0], st_memory = self.stlstm_layer[0](x_gen,
                                                                             hidden_state[0],
                                                                             cell_state[0],
                                                                             st_memory)
            # 对于 mimblock
            for i in range(1, self.num_layers):
                if time_step > 0:
                    if i == 1:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                    else:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = self.stlstm_layer_diff[i - 1](
                            hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])

                preh = hidden_state[i]
                hidden_state[i], cell_state[i], st_memory, mims_cell[i - 1] = self.stlstm_layer[i](  # mimblock
                    hidden_state[i - 1], hidden_state_diff[i - 1], hidden_state[i], cell_state[i],
                    st_memory, mims_cell[i - 1])

            x_gen = self.x_gen(hidden_state[self.num_layers - 1])

            gen_images.append(x_gen)

        gen_images = torch.stack(gen_images, dim=1)

        # gen_images[torch.abs(gen_images) < 0.1] = 0.

        return gen_images


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')
    args = parser.parse_args()
    args.img_channel = 1
    args.num_layers = 4
    args.num_hidden = [12, 12, 12, 12]
    args.input_length = 12
    args.total_length = 24
    args.predict_length = 6
    args.batch_size = 2
    args.img_height = 80
    args.img_width = 120
    args.filter_size = 5
    args.device = 'cpu'
    args.uniform_size = False

    # forward
    inputs = torch.randn(2, args.total_length,
                         args.img_channel, args.img_height, args.img_width)
    mask_true = torch.randn(2, args.total_length - args.input_length - 1,
                            args.img_channel, args.img_height, args.img_width)

    args.img_height = 800
    args.img_width = 1200
    predrnn = MIM(args.num_layers, args.num_hidden, args, norm=True)
    predict = predrnn(inputs, mask_true)
    print(predict.shape)

    inputs = torch.randn(2, args.total_length, args.img_channel, 50, 90)
    mask_true = torch.randn(2, args.total_length - args.input_length - 1, args.img_channel, 50, 90)
    predict = predrnn(inputs, mask_true)
    print(predict.shape)
    cell_list = []

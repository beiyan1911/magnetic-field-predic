import os
import torch
import torch.nn as nn
from torch.optim import Adam
from model import network


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.mix = configs.mix
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'custom': network.MIM
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.MSE_criterion = nn.MSELoss()
        if self.mix:
            self.scaler = torch.cuda.amp.GradScaler()

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train_mode(self):
        for module in self.network.children():
            module.train(True)

    def eval_mode(self):
        for module in self.network.children():
            module.train(False)

    def train(self, frames, mask):
        frames_tensor = frames.to(self.configs.device)
        mask_tensor = mask.to(self.configs.device)
        self.optimizer.zero_grad()
        if self.mix:
            with torch.cuda.amp.autocast():
                next_frames = self.network(frames_tensor, mask_tensor)
                loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # loss.backward()
            # self.optimizer.step()
            return loss.item()
        else:
            next_frames = self.network(frames_tensor, mask_tensor)
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
            loss.backward()
            self.optimizer.step()
            return loss.item()

    def test(self, frames, mask):
        frames_tensor = frames.to(self.configs.device)
        mask_tensor = mask.to(self.configs.device)
        next_frames = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()

import torch.utils.data as data
from torch.utils.data import DataLoader
import glob2
import os
import numpy as np
import torch


class SunspotData(data.Dataset):
    def __init__(self, root, configs):
        self.paths = sorted(glob2.glob(os.path.join(root, '*.npz')))
        _required_length = configs.input_length + configs.predict_length
        if _required_length < configs.total_length:
            self.required_length = _required_length
        else:
            self.required_length = configs.total_length

    def __getitem__(self, index):
        path = self.paths[index]
        all_data = np.load(path)
        sequence = all_data['images'][:self.required_length]
        names = all_data['names'][:self.required_length].tolist()
        return torch.from_numpy(sequence), names

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch Magnetic field Prediction Model')
    parser.add_argument('--input_length', type=int, default=12)
    parser.add_argument('--predict_length', type=int, default=6)
    parser.add_argument('--total_length', type=int, default=24)
    args = parser.parse_args()

    path = '../datasets/train'
    dataloader = DataLoader(dataset=SunspotData(path, configs=args), num_workers=0, batch_size=2, shuffle=True)
    for i in range(5):
        print('======> epoch %d' % i)
        for j, data in enumerate(dataloader):
            print(data.detach().cpu().numpy())

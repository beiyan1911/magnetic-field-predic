import argparse
import os

import datetime
import numpy as np
from torch.utils.data import DataLoader

import model.trainer as trainer
from model.sunspots import SunspotData
from model.model_factory import Model
import torch
from utils.utils import SummaryHelper

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch Magnetic field Prediction Model')

# training/test
parser.add_argument('--is_training', type=bool, default=True)
parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
parser.add_argument('--resume', default=False, help='continue training: True or False')
parser.add_argument('--resume_count', type=int, default=1, help='when resume,from which count to epoch')
parser.add_argument('--is_sunspots', type=bool, default=False, help='Whether it is a sunspot dataset?')
parser.add_argument('--mix', default=False, help='whether use mixed precision')
parser.add_argument('--uniform_size', type=bool, default=False, help='Whether the size of the dataset is uniform?')

# data
parser.add_argument('--dataset_name', type=str, default='sunspots')
parser.add_argument('--train_data_paths', type=str, default='../datasets/MF_datasets/train')
parser.add_argument('--valid_data_paths', type=str, default='../datasets/MF_datasets/valid')
parser.add_argument('--test_data_paths', type=str, default='../datasets/MF_datasets/test')
parser.add_argument('--save_dir', type=str, default='../outputs/MF_results/checkpoints/')
parser.add_argument('--logs_dir', type=str, default='../outputs/MF_results/logs')
parser.add_argument('--gen_frm_dir', type=str, default='../outputs/MF_results/sample')
parser.add_argument('--test_dir', type=str, default='../outputs/MF_results/test')
parser.add_argument('--input_length', type=int, default=12)  # Input sequence length
parser.add_argument('--predict_length', type=int, default=6)  # Prediction sequence length
parser.add_argument('--total_length', type=int, default=24)  # Total length of magnetic field sequence
parser.add_argument('--img_height', type=int, default=60)
parser.add_argument('--img_width', type=int, default=100)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='custom')
parser.add_argument('--pretrained_model', type=str, default='../outputs/MF_results/checkpoints/model.ckpt-1')
parser.add_argument('--num_hidden', type=str, default='48,48,48')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=bool, default=True)
parser.add_argument('--sampling_stop_iter', type=int, default=80)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.0125)

# optimization
parser.add_argument('--num_work', default=5, type=int, help='threads for loading data')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_epoch', type=int, default=120)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=1)
parser.add_argument('--snapshot_interval', type=int, default=1)
parser.add_argument('--num_save_samples', type=int, default=20)

args = parser.parse_args()


def schedule_sampling(eta, epoch):
    zeros = np.zeros((args.batch_size, args.predict_length - 1), dtype=np.float32)
    if not args.scheduled_sampling:
        return 0.0, zeros

    if epoch < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.predict_length - 1))
    true_token = (random_flip < eta)

    real_input_flag = np.zeros(
        (args.batch_size, args.predict_length - 1),
        dtype=np.float32)
    for i in range(args.batch_size):
        for j in range(args.predict_length - 1):
            if true_token[i, j]:
                real_input_flag[i, j] = 1.
            else:
                real_input_flag[i, j] = 0.

    return eta, real_input_flag


def train_wrapper(model):
    # resume train
    resume_count = 1
    if args.resume:
        model.load(args.pretrained_model)
        resume_count = args.resume_count

    # load data
    train_loader = DataLoader(dataset=SunspotData(args.train_data_paths, args), num_workers=args.num_work,
                              batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(dataset=SunspotData(args.valid_data_paths, args), num_workers=0,
                              batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=True)

    train_summary = SummaryHelper(save_path=os.path.join(args.logs_dir, 'train'),
                                  comment='custom', flush_secs=20)
    test_summary = SummaryHelper(save_path=os.path.join(args.logs_dir, 'test'),
                                 comment='custom', flush_secs=20)

    eta = args.sampling_start_value

    for epoch in range(resume_count, args.max_epoch + 1):
        loss = []
        model.train_mode()
        for itr, (imgs, names) in enumerate(train_loader):
            eta, real_input_flag = schedule_sampling(eta, epoch)
            real_input_flag = torch.from_numpy(real_input_flag)
            itr_loss = trainer.train(model, imgs, real_input_flag, args, epoch, itr)
            loss.append(itr_loss)

        train_loss_avg = np.mean(loss)
        train_summary.add_scalar('train/loss', train_loss_avg, global_step=epoch)

        if epoch % args.snapshot_interval == 0:
            model.save(epoch)

        if epoch % args.test_interval == 0:
            model.eval_mode()
            metrics = trainer.test(model, valid_loader, args, epoch, args.gen_frm_dir, args.is_sunspots)
            test_summary.add_scalars('test', metrics, global_step=epoch)


def test_wrapper(model):
    model.load(args.pretrained_model)
    model.eval_mode()
    test_loader = DataLoader(dataset=SunspotData(args.test_data_paths, args), num_workers=0,
                             batch_size=args.batch_size, shuffle=False, pin_memory=False)
    flag = str(datetime.datetime.now().strftime('%Y_%m_%d___%H_%M_%S'))
    trainer.test(model, test_loader, args, flag, args.test_dir, args.is_sunspots)


if __name__ == '__main__':
    print(args)
    torch.manual_seed(2020)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.gen_frm_dir):
        os.makedirs(args.gen_frm_dir)

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    model = Model(args)
    print('====>  Model initialization completed')
    if args.is_training:
        train_wrapper(model)
    else:
        test_wrapper(model)

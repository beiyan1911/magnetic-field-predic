import os.path
import datetime
import cv2
import numpy as np
import torch
import csv
from skimage.metrics import structural_similarity as ski_ssim


def train(model, ims, real_input_flag, configs, epoch, itr):
    cost = model.train(ims, real_input_flag)
    if configs.reverse_input:
        ims_rev = torch.flip(ims, dims=(1,))
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'epoch: ' + str(epoch), 'itr: ' + str(itr),
              'training loss: ' + str(cost))

    return cost


def generator_input_flag(config):
    real_input_flag = torch.from_numpy(np.zeros((config.batch_size, config.predict_length - 1),
                                                dtype=np.float32))
    return real_input_flag


def test(model, valid_loader, configs, epoch, folder, is_sunspots):
    print('====>  ', 'test for epoch  %s' % str(epoch), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    res_path = os.path.join(folder, str(epoch))
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    avg_mse = 0
    img_mse, ssim = [], []
    batch_num = len(valid_loader)
    test_size = batch_num * configs.batch_size
    for i in range(configs.predict_length):
        img_mse.append(0)
        ssim.append(0)

    with torch.no_grad():
        real_input_flag = generator_input_flag(configs)
        for idx, (test_dat, names) in enumerate(valid_loader):
            img_gen = model.test(test_dat, real_input_flag)
            img_gen = img_gen.transpose(0, 1, 3, 4, 2)  # 输出序列 [N S H W C]
            img_seq = test_dat.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)  # 整个输入序列 [N S H W C]
            output_length = configs.predict_length
            img_gen_length = img_gen.shape[1]
            img_out = img_gen[:, -output_length:]

            # MSE per frame
            for i in range(output_length):
                x = img_seq[:, i + configs.input_length, :, :, :]
                gx = img_out[:, i, :, :, :]
                # 对输出数据clip
                if is_sunspots:
                    gx = np.clip(gx, a_min=0, a_max=1.0)
                else:
                    gx = np.clip(gx, a_min=-1., a_max=1.0)
                mse = np.square(x - gx).mean()
                img_mse[i] += mse
                avg_mse += mse

                if is_sunspots:
                    real_frm = np.uint8(x * 255)
                    pred_frm = np.uint8(gx * 255)
                else:
                    real_frm = np.uint8(x * 127.5 + 127.5)
                    pred_frm = np.uint8(gx * 127.5 + 127.5)

                for b in range(configs.batch_size):
                    score, _ = ski_ssim(pred_frm[b], real_frm[b], full=True,
                                        multichannel=True)
                    ssim[i] += score

            # save prediction examples
            if idx <= configs.num_save_samples or (not configs.is_training):
                path = os.path.join(res_path, names[0][0])
                if not os.path.exists(path):
                    os.mkdir(path)
                name_list = [_[0] for _ in names]
                np.savez(os.path.join(res_path, '%s.npz' % names[0][0]), inputs=img_seq[0], preds=img_gen[0],
                         names=name_list)
                for i in range(configs.input_length + configs.predict_length):
                    name = 'gt' + str(i + 1) + '.png'
                    file_name = os.path.join(path, name)
                    if is_sunspots:
                        img_gt = np.uint8(img_seq[0, i, :, :, :] * 255)
                    else:
                        img_gt = np.uint8(img_seq[0, i, :, :, :] * 127.5 + 127.5)

                    cv2.imwrite(file_name, img_gt)
                for i in range(img_gen_length):
                    name = 'pd' + str(i + 2) + '.png'
                    file_name = os.path.join(path, name)
                    img_pd = img_gen[0, i, :, :, :]
                    if is_sunspots:
                        img_pd = np.clip(img_pd, a_min=0, a_max=1.0)
                    else:
                        img_pd = np.clip(img_pd, a_min=-1., a_max=1.0)
                    if is_sunspots:
                        img_pd = np.uint8(img_pd * 255)
                    else:
                        img_pd = np.uint8(img_pd * 127.5 + 127.5)

                    cv2.imwrite(file_name, img_pd)

    avg_mse = avg_mse / test_size
    img_mse = [_ / test_size for _ in img_mse]
    print('mse of avg: ' + str(avg_mse))
    print('mse of seq: ' + str(img_mse))

    ssim = [_ / test_size for _ in ssim]
    avg_ssim = np.mean(ssim)
    print('ssim of avg: ' + str(avg_ssim))
    print('ssim of seq: ' + str(ssim))

    f = open(os.path.join(configs.save_dir, 'Metric.txt'), 'a')
    writer = csv.writer(f, lineterminator='\n')
    metric = [epoch, avg_mse] + img_mse + ssim
    writer.writerow(metric)

    return {'avg_mse': avg_mse, 'avg_ssim': avg_ssim}

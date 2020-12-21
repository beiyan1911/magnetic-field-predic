import numpy as np
from torch.utils.tensorboard import SummaryWriter


class SummaryHelper(object):
    def __init__(self, save_path, comment, flush_secs):
        super(SummaryHelper, self).__init__()
        self.writer = SummaryWriter(log_dir=save_path, comment=comment, flush_secs=flush_secs, filename_suffix='.log')

    def add_summary(self, current_summary, global_step):
        for key, value in current_summary.items():
            if isinstance(value, np.ndarray):
                self.writer.add_image(key, value, global_step)
            elif isinstance(value, float):
                self.writer.add_scalar(key, value, global_step)

    def add_scalar(self, key, value, global_step):
        self.writer.add_scalar(key, value, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_image(self, key, value, global_step):
        self.writer.add_image(key, value, global_step)

    def add_figure(self, key, figure, global_step):
        self.writer.add_figure(key, figure, global_step=global_step)

    @staticmethod
    def get_current_losses(losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        # message = '(epoch: %d, iters: %d ,time: %.3f) ' % (epoch, iters, time)
        message = '  '
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        return message

    def close(self):
        self.writer.close()

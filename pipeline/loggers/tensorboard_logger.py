from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self):
        self.writer = SummaryWriter()

    def log(self, epoch, loss):
        self.writer.add_scalar("Loss (train)", loss, epoch)

    def log_image(self, img_name, img, epoch):
        self.writer.add_image(img_name, img, global_step=epoch)

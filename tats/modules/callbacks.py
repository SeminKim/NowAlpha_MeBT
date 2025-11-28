import torch
import numpy as np
import wandb
from pytorch_lightning.callbacks import Callback
from tats.data import zr_relation, unnormalize_dBZ
from tats.utils import visualize_zr

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        return

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")




class VideoLogger(Callback):
    def __init__(self, batch_frequency, max_videos, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_videos = max_videos
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp


    def log_vid(self, pl_module, batch, batch_idx, split="train"):
        # print(batch_idx, self.batch_freq, self.check_frequency(batch_idx) and hasattr(pl_module, "log_videos") and callable(pl_module.log_videos) and self.max_videos > 0)
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_videos") and
                callable(pl_module.log_videos) and
                self.max_videos > 0):
            # print(batch_idx, self.batch_freq,  self.check_frequency(batch_idx))
            # logger = type(pl_module.logger)
            wandb_logger = pl_module.logger.experiment

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                videos = pl_module.log_videos(batch, split=split, batch_idx=batch_idx)

            vids = [] # t c h w
            for k in videos:
                N = min(videos[k].shape[0], self.max_videos)
                videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = zr_relation(unnormalize_dBZ(videos[k].detach().cpu()))
                    if self.clamp:
                        videos[k] = torch.clamp(videos[k], 0, 20)
                    videos[k] = visualize_zr(videos[k], vmax=10)
                vids.append(videos[k])
                wandb_logger.log({f'{split}/vid_{k}': wandb.Video(videos[k], fps=6)}, step=pl_module.global_step)
            vids = np.concatenate(vids, axis=2)
            wandb_logger.log({f'{split}/vids': wandb.Video(vids, fps=6)}, step=pl_module.global_step)
            if is_train:
                pl_module.train()
        return

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_vid(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_vid(pl_module, batch, batch_idx, split="val")

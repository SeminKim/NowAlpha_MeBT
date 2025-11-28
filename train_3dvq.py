# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tats import VideoData
from tats.modules.callbacks import ImageLogger, VideoLogger
from utils import get_obj_from_str
from omegaconf import OmegaConf

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--base', nargs='*', metavar="base_config.yaml")
    parser.add_argument('--ckpt_path', default=None)

    # parser = VQGAN.add_model_specific_args(parser)
    # parser = VideoData.add_data_specific_args(parser)
    args, unknown = parser.parse_known_args()
    if args.default_root_dir is None:
        args.default_root_dir = ''

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    data = VideoData(config.data)
    data.prepare_data()

    # model = VQGAN(args)
    # configure learning rate
    bs, base_lr = config.data.batch_size, config.exp.base_lr
    ngpu = len(args.gpus.strip(',').split(','))
    if hasattr(config.exp, 'accumulate_grad_batches'):
        args.accumulate_grad_batches = config.exp.accumulate_grad_batches
    accumulate_grad_batches = args.accumulate_grad_batches or 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    if not hasattr(config.exp, 'exact_lr'):
        lr = accumulate_grad_batches * ngpu * bs * base_lr
    else:
        lr = config.exp.exact_lr
    config.model.params.lr = lr

    if config.model.target == 'tats.tats_vqgan.VQGAN':
        config.model.params.spatial_to_channel = config.data.spatial_to_channel
        model = get_obj_from_str(config.model["target"])(config.model.params)
    else:
        model = get_obj_from_str(config.model["target"])(**config.model.params)

    callbacks = []

    if config.model.target == 'tats.tats_vqgan.VQGAN':
        callbacks.append(ModelCheckpoint(monitor='val/recon_loss', save_top_k=3, mode='min', filename='latest_checkpoint'))
        callbacks.append(ModelCheckpoint(every_n_train_steps=50000, save_top_k=-1, filename='{epoch}-{step}-50000-{train/recon_loss:.2f}'))
    else:
        callbacks.append(ModelCheckpoint(monitor='val/rec_loss', save_top_k=3, mode='min', filename='latest_checkpoint'))
        callbacks.append(ModelCheckpoint(every_n_train_steps=50000, save_top_k=-1, filename='{epoch}-{step}-50000-{train/rec_loss:.2f}'))
    callbacks.append(ImageLogger(batch_frequency=1500, max_images=4, clamp=True))
    callbacks.append(VideoLogger(batch_frequency=1500, max_videos=4, clamp=True))

    loggers = []
    loggers.append(WandbLogger(name=args.base[0].split('/')[-1].split('.yaml')[0], entity='kma_vllab', project='KMA_3DVQ'))

    kwargs = dict(gpus=args.gpus,
                  strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True))
    #   strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True))

    # load the most recent checkpoint file
    base_dir = os.path.join(args.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            for fn in os.listdir(ckpt_folder):
                if fn == 'latest_checkpoint.ckpt':
                    ckpt_file = 'latest_checkpoint_prev.ckpt'
                    os.rename(os.path.join(ckpt_folder, fn), os.path.join(ckpt_folder, ckpt_file))
            if len(ckpt_file) > 0:
                args.resume_from_checkpoint = os.path.join(ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s' % args.resume_from_checkpoint)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=args.max_steps,
                                            logger=loggers,
                                            gradient_clip_val=args.gradient_clip_val,
                                            **kwargs)

    trainer.fit(model, data, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()

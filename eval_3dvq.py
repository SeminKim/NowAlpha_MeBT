# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import argparse
import pytorch_lightning as pl
from tats import VideoData
from utils import get_obj_from_str
from omegaconf import OmegaConf
import torch


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--base', nargs='*', metavar="base_config.yaml")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--ignore', action='store_true')
    parser.add_argument('--target_dates', type=str, default=None)

    args, unknown = parser.parse_known_args()
    pl.seed_everything(args.seed)
    if args.default_root_dir is None:
        args.default_root_dir = ''

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    data = VideoData(config.data)
    data.prepare_data()

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
    config.model.params.spatial_to_channel = config.data.spatial_to_channel

    model = get_obj_from_str(config.model["target"])(config.model.params)
    if hasattr(config.model, 'load_path'):
        sd = torch.load(config.model['load_path'], map_location='cpu')['state_dict']
        ign = []
        for k, v in sd.items():
            if 'disc' in k:
                ign.append(k)
        for k in ign:
            del sd[k]

        m, u = model.load_state_dict(sd, strict=False)
    else:
        sd = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
        ign = []
        for k, v in sd.items():
            if 'disc' in k:
                ign.append(k)
        for k in ign:
            del sd[k]
        m, u = model.load_state_dict(sd, strict=False)
    for k in m:
        if 'disc' not in k:
            if 'f1' not in k:
                print(k)
                assert 0

    for k in u:
        if 'disc' not in k:
            if 'f1' not in k:
                print(k)
                assert 0

    kwargs = dict(gpus=args.gpus,
                  strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True))

    trainer = pl.Trainer.from_argparse_args(args,
                                            max_steps=args.max_steps,
                                            **kwargs)
    if args.target_dates is not None:
        data.valset.parse_dataset(args.target_dates, ignore=args.ignore)
    loaders = data._dataloader(False, False, shuffle=False)
    trainer.validate(model, dataloaders=loaders)


if __name__ == '__main__':
    main()

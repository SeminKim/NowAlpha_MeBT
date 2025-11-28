import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tats import VideoData
from utils import get_obj_from_str
from omegaconf import OmegaConf

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():
    pl.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--base', nargs='*', metavar="base_config.yaml")
    parser.add_argument('--ckpt_path', default=None)
    '''
    parser = Net2NetTransformer.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    '''
    args, unknown = parser.parse_known_args()
    if args.default_root_dir is None:
        args.default_root_dir = ''

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    config.data.latent_shape = config.model.mask.params.shape
    assert not config.model['target'].endswith('VToken') or config.data.val_vtoken
    if hasattr(config.model.params, 'use_time_embedding') and config.model.params.use_time_embedding:
        assert config.data.use_time, f'Dataloader should give time information to use time embedding'

    data = VideoData(config.data)

    config.model.params.class_cond_dim = data.n_classes if not config.model.params.unconditional and config.model.params.cond_stage_key == 'label' else None
    config.model.params.latent_shape = config.data.latent_shape
    print(f'assuming latent shape {config.model.params.latent_shape}')
    model = get_obj_from_str(config.model["target"])(config.model.params, config.model.vqvae,
                                                     config.model.mask, cond_stage_key=config.model.params.cond_stage_key)
    if config.model["target"] == "tats.COMMIT_transformer.Net2NetTransformerVToken":
        model.cond_latent_frames = config.data.before // config.data.input_interval

    callbacks = []
    if not config.model["target"] == "tats.COMMIT_transformer.Net2NetTransformerVToken":
        callbacks.append(ModelCheckpoint(every_n_epochs=1, save_top_k=-1, filename='{epoch}-{step}-{train/loss:.2f}'))
    else:
        callbacks.append(ModelCheckpoint(every_n_epochs=1, save_top_k=-1, filename='{epoch}-{step}-{train/loss:.2f}'))

    loggers = []
    loggers.append(WandbLogger(name=args.base[0].split('/')[-1].split('.yaml')[0], project='Climate_3dVQ', entity='kma_vllab'))

    kwargs = dict(gpus=args.gpus,
                  strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False))

    # configure learning rate
    bs, base_lr = config.data.batch_size, config.exp.base_lr
    ngpu = len(args.gpus.strip(',').split(','))
    if hasattr(config.exp, 'accumulate_grad_batches'):
        args.accumulate_grad_batches = config.exp.accumulate_grad_batches
    accumulate_grad_batches = args.accumulate_grad_batches or 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    if not hasattr(config.exp, 'exact_lr'):
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    else:
        model.learning_rate = config.exp.exact_lr
    if not hasattr(config.exp, 'warmup_steps'):
        model.warmup_steps = 0
    else:
        model.warmup_steps = config.exp.warmup_steps
    if not hasattr(config.exp, 'weight_decay'):
        model.weight_decay = 0.01
    else:
        model.weight_decay = config.exp.weight_decay
    if hasattr(config.exp, 'cosine_lr'):
        model.cosine_lr = config.exp.cosine_lr
    else:
        model.cosine_lr = False
    print(f'LR: {model.learning_rate}, WD: {model.weight_decay}')

    # load the most recent checkpoint file
    base_dir = os.path.join(args.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir) and args.ckpt_path is None:
        print(f"BASE DIR ({base_dir}) exists. Find recent checkpoints...")
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id >= version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            if os.path.exists(ckpt_folder):
                for fn in os.listdir(ckpt_folder):
                    if fn == 'latest_checkpoint.ckpt':
                        ckpt_file = 'latest_checkpoint.ckpt'
                        # os.rename(os.path.join(ckpt_folder, fn), os.path.join(ckpt_folder, ckpt_file))
                if len(ckpt_file) > 0:
                    args.ckpt_path = os.path.join(ckpt_folder, ckpt_file)
                    print('will start from the recent ckpt %s' % args.ckpt_path)
                else:
                    print('Could not find the checkpoint')
        else:
            print(f"Could not find the log folder")
    if args.ckpt_path is not None:
        print('will start from the recent ckpt %s' % args.ckpt_path)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=args.max_steps,
                                            logger=loggers,
                                            **kwargs)
    trainer.fit(model, data, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()

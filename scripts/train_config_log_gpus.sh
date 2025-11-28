#!/bin/bash

python train_transformer.py \
        --base $1 \
        --default_root_dir $2 \
        --gpus $3 \
        --check_val_every_n_epoch=3 \
        --max_steps 20000000 \
        --accumulate_grad_batches 1 \
        --limit_val_batches 0.01 \
        --gradient_clip_val 1.0 \
        --num_sanity_val_steps 0

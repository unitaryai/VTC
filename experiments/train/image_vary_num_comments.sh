#!/usr/bin/env bash

# Train models with different number of comments

device=0

for n_comms in 1 3 5 7 9 ; do

    if [ $n_comms -ge 5 ]; then
        device=1
    fi

    echo "Training exp with $n_comms comments on device $device"

    python train.py --config "configs/pretrained_clip_comments_attn_frozen.jsonc" \
                    --branch_to_adapt text \
                    --save_dir "varying_comm_experiments" \
                    --exp_name "frozen_pretrained_clip_${n_comms}_comments" \
                    --num_comms $n_comms --batch_size 128 \
                    --device $device --epochs 12 \
                    --cached_vision_features "./clip_vit_embeddings.pth" &

done

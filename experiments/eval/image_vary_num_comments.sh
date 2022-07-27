#!/usr/bin/env bash

# Evaluate models with different number of comments

exp_root="varying_comm_experiments/models/frozen_pretrained_clip"
date="005_May20_16:08"
epoch_num=10
bs=80
device=1

for n_comm in 1 3 5 7 9; do
    
    if [ $n_comm -ge 5 ]; then
        device=3
    fi

    exp="${exp_root}_${n_comm}_comments"
    
    python evaluate.py \
        --config "${exp}/${date}/config.json" \
        --resume "${exp}/${date}/checkpoint-epoch${epoch_num}.pth" \
        --device $device --branch_to_adapt_val text --batch_size $bs &

done 
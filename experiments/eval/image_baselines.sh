#!/usr/bin/env bash

# zero shot clip
python evaluate.py --config configs/pretrained_clip.jsonc \
                   --device 0 \
                   --batch_size 128 &

# zero shot clip with averaging
python evaluate.py --config configs/pretrained_clip.jsonc \
                   --device 0 --add_comments always --num_comms 5 --num_imlabels 0 \
                   --batch_size 128 --comment_fusion averaging &

# finetuned clip with averaging, eval on title
python evaluate.py --config configs/pretrained_clip.jsonc \
                --add_comments train_only \
                --device 0 \
                --batch_size 128 --comment_fusion None \
                --resume saved/models/pretrained_clip_averaging_comments/005_May23_01:54/checkpoint-epoch10.pth &

# finetuned clip with averaging, eval with averaging
python evaluate.py --config configs/pretrained_clip.jsonc \
                --add_comments always \
                --num_comms 5 --num_imlabels 0 --device 0 \
                --batch_size 128 --comment_fusion averaging \
                --resume saved/models/pretrained_clip_averaging_comments/005_May23_01:54/checkpoint-epoch10.pth &

# finetuned adapted
exp_root=saved/models/frozen_pretrained_clip_fix_nowikidesc
date=001_May20_16:44
epoch_num=10
bs=80

for eval_br in image text; do
    python evaluate.py \
        --config ${exp_root}_${exp}/${date}/config.json \
        --resume ${exp_root}_${exp}/${date}/checkpoint-epoch${epoch_num}.pth \
        --device 0 --branch_to_adapt_val $eval_br --batch_size $bs &

done
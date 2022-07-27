#!/usr/bin/env bash

# clip baseline
python train.py --config "configs/pretrained_clip.jsonc" \
                --add_comments never --comment_fusion None \
                --exp_name "pretrained_clip_title_only" \
                --device 3 --epochs 11 \
                --batch_size 50 &

# finetune clip + avg title&comments
python train.py --config "configs/pretrained_clip.jsonc" \
                --add_comments always --comment_fusion averaging \
                --exp_name "pretrained_clip_averaging_comments" \
                --num_comms 5 --device 0  --epochs 11 \
                --batch_size 50 &

# frozen adapted title/image with comments
for br in text image; do     
    python train.py --config "configs/pretrained_clip_comments_attn_frozen.jsonc" \
                    --branch_to_adapt "$br" --branch_to_adapt_val "$br" \
                    --exp_name "frozen_clip_comments_${br}_branch" \
                    --num_comms 5 --device 1  --epochs 12 \
                    --cached_vision_features "./clip_vit_embeddings.pth" &
done

# finetune adapted title/image with comments
for br in text image; do     
    python train.py --config "configs/pretrained_clip_comments_attention.jsonc" \
                    --branch_to_adapt "$br" --branch_to_adapt_val "$br" \
                    --exp_name "finetuned_clip_comments_${br}_branch" \
                    --num_comms 5 --device 2  --epochs 22 \
                    --cached_vision_features "./clip_vit_embeddings.pth" \
                    --resume "saved/models/frozen_clip_comments_${br}_branch/002_Jul15_01:42/checkpoint-epoch12.pth" &
done

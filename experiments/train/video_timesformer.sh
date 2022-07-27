python train.py -c configs/pretrained_clip_timesformer_comments_attention.jsonc \
            --num_comms 5 --num_imlabels 0 -d 3 \
            --residual_activation none \
            --epochs 1 \
            --freeze none \
            --save_dir "/checkpoints/timesformer" \
            --exp_name "timesformer" \
            --branch_to_adapt image --branch_to_adapt_val image \
            --visual_device "cuda:1"

python train.py -c configs/pretrained_clip_timesformer_comments_attention.jsonc \
            --num_comms 5 --num_imlabels 0 -d 3 \
            --residual_activation none \
            --epochs 1 \
            --freeze none \
            --save_dir "/checkpoints/timesformer" \
            --exp_name "timesformer_adapt_text" \
            --branch_to_adapt text --branch_to_adapt_val text \
            --visual_device "cuda:0"


python train.py -c configs/pretrained_clip_1frame_comments_attention.jsonc \
            --num_comms 5 --num_imlabels 0 -d 0 \
            --residual_activation none \
            --epochs 1 \
            --freeze none \
            --save_dir "/checkpoints/timesformer" \
            --exp_name "1frame" \
            --branch_to_adapt image --branch_to_adapt_val image


python train.py -c configs/pretrained_clip_1frame_comments_attention.jsonc \
            --num_comms 5 --num_imlabels 0 -d 1 \
            --residual_activation none \
            --epochs 1 \
            --freeze none \
            --save_dir "/checkpoints/timesformer" \
            --exp_name "1frame_text_branch" \
            --branch_to_adapt text --branch_to_adapt_val text


python train.py -c configs/pretrained_clip_timesformer_comments_attention.jsonc \
            -r "/checkpoints/timesformer/models/1frame/001_Mar05_22:16/checkpoint-epoch1.pth" \
            --num_comms 5 --num_imlabels 0 -d 3 \
            --residual_activation none \
            --epochs 2 \
            --freeze none \
            --save_dir "/checkpoints/timesformer" \
            --exp_name "timesformer_ft_from_oneframe" \
            --branch_to_adapt image --branch_to_adapt_val image \
            --visual_device "cuda:1"

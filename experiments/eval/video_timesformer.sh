TIMESFORMER_IMAGE_BRANCH="/checkpoints/timesformer/models/timesformer/004_Mar05_20:12/checkpoint-epoch1.pth"
TIMESFORMER_IMAGE_BRANCH_FT="/checkpoints/timesformer/models/timesformer_ft_from_oneframe/003_Mar06_04:38/checkpoint-epoch2.pth"
TIMESFORMER_TEXT_BRANCH="/checkpoints/timesformer/models/timesformer_adapt_text/002_Mar06_06:23/checkpoint-epoch1.pth"

ONEFRAME_IMAGE_BRANCH="/checkpoints/timesformer/models/1frame/001_Mar05_22:16/checkpoint-epoch1.pth"
ONEFRAME_TEXT_BRANCH="/checkpoints/timesformer/models/1frame_text_branch/001_Mar06_19:49/checkpoint-epoch1.pth"

res_dir="/checkpoints/timesformer/"

# Reddit
# 5000 vids

python retrieval_evaluation.py --dataset Reddit_videos --first_chunk_only --checkpoint "$TIMESFORMER_IMAGE_BRANCH" --model_type clip_timesformer_finaltf --device "cuda:1" --split test --branch_to_adapt skip --out_csv "$res_dir/timesformer_image_branch_skipadapt.csv"
python retrieval_evaluation.py --dataset Reddit_videos --first_chunk_only --checkpoint "$TIMESFORMER_IMAGE_BRANCH" --model_type clip_timesformer_finaltf --device "cuda:1" --split test --branch_to_adapt image --out_csv "$res_dir/timesformer_image_branch_imageadapt.csv"

python retrieval_evaluation.py --dataset Reddit_videos --first_chunk_only --checkpoint "$TIMESFORMER_IMAGE_BRANCH_FT" --model_type clip_timesformer_finaltf --device "cuda:3" --split test --branch_to_adapt skip --out_csv "$res_dir/timesformer_image_branchft_skipadapt.csv" &
python retrieval_evaluation.py --dataset Reddit_videos --first_chunk_only --checkpoint "$TIMESFORMER_IMAGE_BRANCH_FT" --model_type clip_timesformer_finaltf --device "cuda:3" --split test --branch_to_adapt image --out_csv "$res_dir/timesformer_image_branchft_imageadapt.csv" &

python retrieval_evaluation.py --dataset Reddit_videos --first_chunk_only --checkpoint "$TIMESFORMER_TEXT_BRANCH" --model_type clip_timesformer_finaltf --device "cuda:1" --split test --branch_to_adapt skip --out_csv "$res_dir/timesformer_text_branch_skipadapt.csv" &
python retrieval_evaluation.py --dataset Reddit_videos --first_chunk_only --checkpoint "$TIMESFORMER_TEXT_BRANCH" --model_type clip_timesformer_finaltf --device "cuda:1" --split test --branch_to_adapt text --out_csv "$res_dir/timesformer_text_branch_textadapt.csv" &

python retrieval_evaluation.py --dataset Reddit_videos --first_frame_only --checkpoint "$ONEFRAME_IMAGE_BRANCH" --model_type pretrained_clip_finaltf --device "cuda:3" --split test --branch_to_adapt skip --out_csv "$res_dir/oneframe_image_branch_skipadapt.csv" &
python retrieval_evaluation.py --dataset Reddit_videos --first_frame_only --checkpoint "$ONEFRAME_IMAGE_BRANCH" --model_type pretrained_clip_finaltf --device "cuda:3" --split test --branch_to_adapt image --out_csv "$res_dir/oneframe_image_branch_imageadapt.csv" &

python retrieval_evaluation.py --dataset Reddit_videos --first_frame_only --checkpoint "$ONEFRAME_TEXT_BRANCH" --model_type pretrained_clip_finaltf --device "cuda:3" --split test --branch_to_adapt skip --out_csv "$res_dir/oneframe_text_branch_skipadapt.csv" &
python retrieval_evaluation.py --dataset Reddit_videos --first_frame_only --checkpoint "$ONEFRAME_TEXT_BRANCH" --model_type pretrained_clip_finaltf --device "cuda:3" --split test --branch_to_adapt text --out_csv "$res_dir/oneframe_text_branch_textadapt.csv" &

## K700
# 6292 vids

python retrieval_evaluation.py --dataset K700_videos --first_chunk_only --checkpoint "$TIMESFORMER_IMAGE_BRANCH_FT" --model_type clip_timesformer_finaltf --device "cuda:3" --split test --branch_to_adapt skip --out_csv "$res_dir/k700_timesformer_image_branchft_skipadapt.csv" &
python retrieval_evaluation.py --dataset K700_videos --first_chunk_only --checkpoint "$TIMESFORMER_IMAGE_BRANCH_FT" --model_type clip_timesformer_finaltf --device "cuda:3" --split test --branch_to_adapt image --out_csv "$res_dir/k700_timesformer_image_branchft_imageadapt.csv" &

python retrieval_evaluation.py --dataset K700_videos --first_frame_only --checkpoint "$ONEFRAME_IMAGE_BRANCH" --model_type pretrained_clip_finaltf --device "cuda:3" --split test --branch_to_adapt skip --out_csv "$res_dir/k700_oneframe_image_branch_skipadapt.csv" &
python retrieval_evaluation.py --dataset K700_videos --first_frame_only --checkpoint "$ONEFRAME_IMAGE_BRANCH" --model_type pretrained_clip_finaltf --device "cuda:3" --split test --branch_to_adapt image --out_csv "$res_dir/k700_oneframe_image_branch_imageadapt.csv" &

#python retrieval_evaluation.py --dataset K700_videos --first_frame_only --model_type pretrained_clip --device "cuda:0" --split test --out_csv "$res_dir/k700_clip.csv"

# Livebot
# 100 vids

python retrieval_evaluation.py --dataset livebot --first_chunk_only --checkpoint "$TIMESFORMER_IMAGE_BRANCH_FT" --model_type clip_timesformer_finaltf --device "cuda:3" --split test --branch_to_adapt skip --out_csv "$res_dir/livebot_timesformer_image_branchft_skipadapt.csv" &
python retrieval_evaluation.py --dataset livebot --first_chunk_only --checkpoint "$TIMESFORMER_IMAGE_BRANCH_FT" --model_type clip_timesformer_finaltf --device "cuda:0" --split test --branch_to_adapt image --out_csv "$res_dir/livebot_timesformer_image_branchft_imageadapt.csv" &

python retrieval_evaluation.py --dataset livebot --first_frame_only --checkpoint "$ONEFRAME_IMAGE_BRANCH" --model_type pretrained_clip_finaltf --device "cuda:0" --split test --branch_to_adapt skip --out_csv "$res_dir/livebot_oneframe_image_branch_skipadapt.csv" &
python retrieval_evaluation.py --dataset livebot --first_frame_only --checkpoint "$ONEFRAME_IMAGE_BRANCH" --model_type pretrained_clip_finaltf --device "cuda:1" --split test --branch_to_adapt image --out_csv "$res_dir/livebot_oneframe_image_branch_imageadapt.csv" &

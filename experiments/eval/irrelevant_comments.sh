#!/usr/bin/env bash
bs=80
device=1
save_dir="results_irrelevant_n_comments"

mkdir $save_dir
for n_comm_eval in 0 1 3 5 7 11; do

    if [ $n_comm_eval -ge 3 ]; then
        device=3
    fi
    python evaluate.py \
        --config "${exp::-22}/config.json" \
        --eval_name "averaging_trained_on_5_comments_${n_comm_eval}_irrelevant" \
        --batch_size $bs --comment_fusion "averaging" --num_comms 5 --add_comments "always"   \
        --resume $AVG_CKPT \
        --save_dir $save_dir \
        --test_on_over_k_comms 3 --test_set_limit 5000 \
        --num_irrelevant_comments $n_comm_eval \
        --device $device &

done

for n_comm_eval in 0 1 3 5 7 11; do

    if [ $n_comm_eval -ge 3 ]; then
        device=3
    fi
    python evaluate.py \
        --config "${exp::-22}/config.json" \
        --eval_name "adapted_title_trained_on_5_comments_${n_comm_eval}_irrelevant" \
        --num_comms 5 --branch_to_adapt_val text --add_comments "always" \
        --batch_size $bs \
        --resume $ADAPT_TITLE_CKPT \
        --save_dir $save_dir \
        --test_on_over_k_comms 3 --test_set_limit 5000 \
        --num_irrelevant_comments $n_comm_eval \
        --device $device  &
done 

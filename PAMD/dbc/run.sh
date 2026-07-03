#!/bin/bash

for SEED in 51 52 53 54
do
  echo "Running with seed $SEED"

  python train.py \
    --domain_name cheetah \
    --task_name run \
    --resource_files 'distractors/*.mp4' \
    --img_source 'video' \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --num_train_steps 1010000 \
    --work_dir ./log/cheetah_seed_$SEED \
    --seed $SEED \
    --agent BisimOursAgent

done

echo "All runs completed."
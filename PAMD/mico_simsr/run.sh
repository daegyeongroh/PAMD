DOMAIN=cheetah
TASK=run

for SEED in 1 2 3
do
    MUJOCO_GL="glfw" CUDA_VISIBLE_DEVICES=0 python -u train.py \
        --domain_name ${DOMAIN} \
        --task_name ${TASK} \
        --encoder_type pixel \
        --action_repeat 4 \
        --pre_transform_image_size 84 \
        --image_size 84 \
        --work_dir ./tmp \
        --agent mico_simsr_ours_sac \
        --frame_stack 3 \
        --seed ${SEED} \
        --critic_lr 1e-3 \
        --actor_lr 1e-3 \
        --eval_freq 10000 \
        --batch_size 128 \
        --save_video \
        --num_train_steps 1010000 \
        > ${DOMAIN}_${TASK}_${SEED}.log
done
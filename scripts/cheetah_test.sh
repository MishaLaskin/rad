CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel --work_dir ./tmp/translation \
    --action_repeat 4 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 108 \
    --agent rad_sac --frame_stack 3 --data_augs translate  \
    --seed 1208 --critic_lr 2e-4 --actor_lr 2e-4 --eval_freq 10000 \
    --batch_size 128 --num_train_steps 600000 --init_steps 10000 \
    --num_filters 32 --encoder_feature_dim 64  --replay_buffer_capacity 100000 \

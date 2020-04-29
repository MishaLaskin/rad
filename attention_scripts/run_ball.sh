#for data_aug in  crop no_aug cutout cutout_color flip rotate color_jitter
for data_aug in gray
    do  
        python train.py --domain_name ball_in_cup --task_name catch --encoder_type pixel --pre_transform_image_size 100 --image_size 84 --work_dir ./results --agent rad_sac --frame_stack 3 --seed 1234 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --num_train_steps 250000 --batch_size 128 --action_repeat 4 --save_model --data_augs $data_aug
    done

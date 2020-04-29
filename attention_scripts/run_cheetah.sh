for data_aug in  gray
#for data_aug in  crop no_aug cutout cutout_color flip rotate color_jitter
    do  
        python train.py --domain_name cheetah --task_name run --encoder_type pixel --pre_transform_image_size 100 --image_size 84 --work_dir ./results --agent rad_sac --frame_stack 3 --seed 1234 --critic_lr 2e-4 --actor_lr 2e-4 --eval_freq 10000 --encoder_lr 2e-4 --num_train_steps 125000 --batch_size 128  --action_repeat 4 --data_augs $data_aug --save_model
    done

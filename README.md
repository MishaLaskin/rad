# Reinforcement Learning with Augmented Data (RAD)

Official codebase for [Reinforcement Learning with Augmented Data](https://mishalaskin.github.io/rad). This codebase was originally forked from [CURL](https://mishalaskin.github.io/curl). 

Additionally, here is the [codebase link for ProcGen experiments](https://github.com/pokaxpoka/rad_procgen).


## BibTex

```
@unpublished{laskin_lee2020rad,
  title={Reinforcement Learning with Augmented Data},
  author={Laskin, Michael and Lee, Kimin and Stooke, Adam and Pinto, Lerrel and Abbeel, Pieter and Srinivas, Aravind},
  note={arXiv:2004.14990}
}
```

## Installation 

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

## Instructions
To train a RAD agent on the `cartpole swingup` task from image-based observations run `bash script/run.sh` from the root of this directory. The `run.sh` file contains the following command, which you can modify to try different environments / augmentations / hyperparamters.

```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel --work_dir ./tmp/cartpole \
    --action_repeat 8 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 84 \
    --agent rad_sac --frame_stack 3 --data_augs flip  \
    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 200000 &
```

## Data Augmentations 

Augmentations can be specified through the `--data_augs` flag. This codebase supports the augmentations specified in `data_augs.py`. To chain multiple data augmentation simply separate the augmentation strings with a `-` string. For example to apply `crop -> rotate -> flip` you can do the following `--data_augs crop-rotate-flip`. 

All data augmentations can be visualized in `All_Data_Augs.ipynb`. You can also test the efficiency of our modules by running `python data_aug.py`.


## Logging 

In your console, you should see printouts that look like this:

```
| train | E: 13 | S: 2000 | D: 9.1 s | R: 48.3056 | BR: 0.8279 | A_LOSS: -3.6559 | CR_LOSS: 2.7563
| train | E: 17 | S: 2500 | D: 9.1 s | R: 146.5945 | BR: 0.9066 | A_LOSS: -5.8576 | CR_LOSS: 6.0176
| train | E: 21 | S: 3000 | D: 7.7 s | R: 138.7537 | BR: 1.0354 | A_LOSS: -7.8795 | CR_LOSS: 7.3928
| train | E: 25 | S: 3500 | D: 9.0 s | R: 181.5103 | BR: 1.0764 | A_LOSS: -10.9712 | CR_LOSS: 8.8753
| train | E: 29 | S: 4000 | D: 8.9 s | R: 240.6485 | BR: 1.2042 | A_LOSS: -13.8537 | CR_LOSS: 9.4001
```
The above output decodes as:

```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
A_LOSS - average loss of actor
CR_LOSS - average loss of critic
```

All data related to the run is stored in the specified `working_dir`. To enable model or video saving, use the `--save_model` or `--save_video` flags. For all available flags, inspect `train.py`. To visualize progress with tensorboard run:

```
tensorboard --logdir log --port 6006
```

and go to `localhost:6006` in your browser. If you're running headlessly, try port forwarding with ssh.


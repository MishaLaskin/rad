# Reinforcement Learning with Augmented Data (RAD)

---
**NOTE**

The instructions for running code for the original Data Augmentations paper is given below. For instructions related to the CS 7643 Fall 2020 Final Project, please find the header [CS 7643 Fall 2020 Project Reproducibility Instructions](#cs-7643-fall-2020-project-reproducibility-instructions) below.

---

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

# CS 7643 Fall 2020 Project Reproducibility Instructions

## Testing  Generalization  and  Performancewith More Data Augmentations
To run this, you should first be in the rad folder and then open the run.sh script present in the scripts folder. In that script '--data_augs' flag can be changed to the follwong 4 options to run the 4 newly implemented augmentation techniques:
1. rgb_shift - Randomly shift rgb values for images
2. channel_shuffle - Shuffle RGB values between channels
3. median_blur - Apply 3x3 median filter to image
4. rand_inv - Randomly select some subset of batch and invert colors 

After selecting the augmentation technique you like, simply run the file through terminal.

## Data Augmentation with GANs
### To train the GAN model:
1. Run the script - run_gan.sh with the following flags being
* '--data_augs' as no_aug 
* '--num_train_steps' as minimum as 10000 
* 'save_buffer' as True
2. This will create a bunch of observations which can be found under 'buffer' directory.
3. You can now open 'scripts/run_gan.sh'. The script calls the 'train_gan.py' file which trains generator and critic models. There are a number of flags that can be set much some of them needs to be a specific value as mentioned below
* '--domain_name' - This variable sets the game that is played. It is currently set to 'cartpole', which is what the tests for the project were conducted on.
* '--task_name' - This variable sets the task. Again, our project tests ran on 'swingup'.
* '--work_dir' - It is specified as './tmp'. This can be changed, but you must create the folder that is specified here (including 'tmp').
* '--seed' - This is the seed that the projects were tested on is 23.
* '--n_epochs' - This is the number of epochs to train on and was 400 for the project.
* '--z_dim' - This is the dimension of the noise vector and is set to 64. If changed then corresponding generator and discriminator models must be changed.
* '--display_step' - This if for visualization of the generated images
* '--batch_size' - This is the batch size, this was set to 4
* '--lr' - This is learning rate set to 0.0002
* '--beta_1' - This is a momentum tern set to 0.5
* '--beta_2' - This is a momentum tern set to 0.999
* '--c_lambda' - This is weight of regularization set to 10
* '--crit_repeats' - This is the number of times to update thecritic per generator update and set to 5
* '--device' - set to cuda
* '--display' - This is whether to display the generated images and loss vs step plot
* '--buffer_for_gan' - This is the observation file that you want the GAN to train on which is of the form './tmp/cartpole-swingup-11-17-im84-b128-s23-pixel/buffer/80000_90000.pt'
4. At the end you will have generator and critic/discriminator models stored in directories such as './tmp/cartpole-swingup-11-29-b4-s23/gan_gen' and './tmp/cartpole-swingup-11-29-b4-s23/gan_disc'.

### To  train the agent:
1. Run the script - run.sh with the following flags as - 
* '--encoder_type' as pixel 
* '--pre_transform_image_size' as 100
* '--image_size' as 100
* '--frame_stack' as 1 
* '--data_augs* as gan
* '--model_gen_dir' as the name of the generator model that you would want to use such as './tmp/cartpole-swingup-11-29-b4-s23/gan_gen/gan_gen_3.pt'
* '-z_dim' is defaluted to 64 but it should be same as what was used to train GAN above
2. Other flag options can be used as is suitable.


## Instructions to Run a Trained Model against Adversarial Observations
This step assumes that you have a trained model, ready to run. It assumes that you've saved both the actor and critic. One can do this by running the training as instructed in the main [instructions](#instructions) section at the top and setting the save_model flag and choosing the eval_freq. The models for the project were trained for 30,000 steps, saving every 5000 steps. We used seed 23 on the cartpole swingup task training the RadSac agent.

Once you have saved the model, you can open 'scripts/run_tests.sh'. The script calls the 'adversarial_obs.py' file. There are a number of flags that can be modified. You'll want to specify the following:

* '--domain_name' - This variable sets the game that is played. It is currently set to 'cartpole', which is what the tests for the project were conducted on.
* '--task_name' - This variable sets the task. Again, our project tests ran on 'swingup'.
* '--work_dir' - It is specified as './test'. This can be changed, but you must create the folder that is specified here (including 'test').
* '--seed' - This is the seed that the projects were tested on is 54.
* '--load_step' - This variable is the only one sent in via an argument. It loads the model trained to the corresponding step. For the project, we tested on models traned to between 10,000 and 30,000 steps, with 5,000 step increments.
* '--eval_steps' - This variable denotes the number of evaluation steps to run. An evaluation step consists of some number of episodes.
* '--num_eval_episodes' - This variable tells us how many episodes should run per step.
* '--train_data_augs' - This variable should be set to the data augmentation used when training.
* '--attack_prob' - This variable denotes the adversarial attack probability. The observation is modified to an adversarial one at a rate approximately equivalent to this probability.
* '--adversarial_iters' - This variable denotes the number of iterations that the adversarial gradient ascent on the observations runs for. The number used for the project was 10.
* '--train_dir' - This variable should point to the top level model directory inside 'tmp'. This should be created during training.

The above variables are already set per the original project parameters. They can be left alone and the script can be directly called to replicate the project. The only ones you may need to modify are 'train_dir', 'work_dir', and 'train_data_augs'. The terminal output gives the evaluation results. The main directory corresponding to the test is saved in the 'test' directory, with some of the parameter settings above identifying it. Within this directory, images of observations and the correponding adversarial observations are stored in the 'image' subdirectory. A video of the episodes are saved in the 'video' subdirectory.

## Training the LSTM on the Cartpole Swingup Task
First, we go over training the LSTM. You should call 'scripts/run_lstm.sh' to train the LSTM. This file calls 'train_lstm.py' with some parameters. Again, the script can be directly called as it is, to replicate the project. We walk through some of the parameters to give the reader familiarity.

* '--domain_name' - As above, we run on the 'cartpole' domain.
* '--task_name' - As above, we run on the 'swingup' task.
* '--work_dir' - In this case, since we are training a model, we give the 'tmp' directory. This is in line with the main repo. All models are stored in './tmp'.
* '--data_augs' - We pass in 'no_augs'. The LSTM for the project was trained on no data augmentations.
* '--seed' - The seed we set for LSTM training was 94.
* '--critic_lr/--actor_lr' - The learning rate for both (after much hyperparameter tuning) was set to 1.5e-5, which was found to produce a good training regime.
* '--eval_freq' - In this case, we saved and evaluated our model every 100 steps.
* '--num_train_steps' - We only run our model to 1500 steps, since the LSTM is heavier to train.
* '--lstm_num_layers' - The number of layers in the LSTM was set as 1.
* '--lstm_dropout' - The dropout rate for the LSTM, which was irrelevant since the number of layers was 1.
* '--lstm_lookback' - This is the number of episodes that the LSTM looks back when training. The LSTM only looks back to a maximum of 5 episodes.

As stated above, the script as is can be run to reproduce the project. Once the LSTM is trained to 1,500 steps you can proceed to the next section.

## Evaluating the LSTM against Adversarial Observations
Next, we evaluate the LSTM against Adversarial Observations to see how it performs in comparison to the MLP based model trained on Data Augmentations. You can call 'scripts/run_tests_lstm.sh' to evaluate the LSTM model. Almost all of the parameters are the same as in the section [Instructions to Run a Trained Model against Adversarial Observations](#instructions-to-run-a-trained-model-against-adversarial-observations). A few changes are given below:

* '--num_eval_episodes' - This parameter should be changed to 5, to replicate the project.
* '--seed' - We tested the LSTM using seed 59.
* '--eval_steps' - We only evaluate the LSTM on 10 steps, instead of 15 as it was for the MLP models.
* '--lstm_lookback' - We restrict the lookback of the LSTM when running tests to just 5 steps prior, instead of the entire episode history.

Again, the script can be run as it is in the github repo to reproduce the project. Everything else is the same as in the previous section about Adversarial Observation tests.

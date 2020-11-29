import argparse
import numpy as np
from torchvision import transforms
import tensorflow as tf
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
import utils
import time
import os
import json
from gan_generator import Generator
from gan_critic import Critic

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='swingup')

    parser.add_argument('--n_epochs', default=400, type=int)
    parser.add_argument('--z_dim', default=64, type=int)
    parser.add_argument('--display_step', default=1000, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--beta_1', default=0.5, type=float)
    parser.add_argument('--beta_2', default=0.999, type=float)
    parser.add_argument('--c_lambda', default=10, type=float)
    parser.add_argument('--crit_repeats', default=5, type=int)

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--display', default=False, type=bool)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--model_gen_dir', default='gan_gen', type=str)
    parser.add_argument('--model_disc_dir', default='gan_disc', type=str)
    parser.add_argument('--buffer_for_gan', default='./tmp/cartpole-swingup-11-17-im84-b128-s23-pixel/buffer/80000_90000.pt', type=str)

    args = parser.parse_args()
    return args

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images
    mixed_images = real * epsilon + fake * (1 - epsilon)


    # critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, this function calculates the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty

def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    gen_loss = - torch.mean(crit_fake_pred)
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    crit_loss =  - torch.mean(crit_real_pred) + torch.mean(crit_fake_pred) + c_lambda*gp
    return crit_loss
     

def main():
    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)

    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed) 
    args.work_dir = args.work_dir + '/'  + exp_name

    utils.make_dir(args.work_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_gen_dir = utils.make_dir(os.path.join(args.work_dir, args.model_gen_dir))
    model_disc_dir = utils.make_dir(os.path.join(args.work_dir, args.model_disc_dir))

    gen = Generator(args.z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
    crit = Critic().to(device) 
    crit_opt = torch.optim.Adam(crit.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)

    cur_step = 0
    generator_losses = []
    critic_losses = []
    buf = torch.load(args.buffer_for_gan)
    replay_buffer = buf[0]
    replay_buffer = replay_buffer[:, :3, :, :]
    no_of_batches = int(replay_buffer.shape[0]/args.batch_size)
    for epoch in range(args.n_epochs):
        for i in range(0, no_of_batches):
            if(i!=no_of_batches):
                real = torch.tensor(replay_buffer[i*args.batch_size:(i+1)*args.batch_size, :, :, :]/255.).float()
            else:
                real = torch.tensor(replay_buffer[i*args.batch_size:, :, :, :]/255.).float()
            cur_batch_size = len(real)
            real = real.to(device)
            mean_iteration_critic_loss = 0
            for _ in range(args.crit_repeats):
                crit_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, args.z_dim, device=device)
                fake = gen(fake_noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)
                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(crit, real, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, args.c_lambda)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_loss += crit_loss.item() / args.crit_repeats
                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()
            critic_losses += [mean_iteration_critic_loss]

            ### Update generator ###
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, args.z_dim, device=device)
            fake_2 = gen(fake_noise_2)
            crit_fake_pred = crit(fake_2)

            gen_loss = get_gen_loss(crit_fake_pred)
            gen_loss.backward()

            # Update the weights
            gen_opt.step()

            # Keep track of the average generator loss
            generator_losses += [gen_loss.item()]

            ### Visualization code  - uncomment for visualization###
            if cur_step % args.display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-args.display_step:]) / args.display_step
                crit_mean = sum(critic_losses[-args.display_step:]) / args.display_step
                print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
                if args.display :
                    show_tensor_images(fake)
                    show_tensor_images(real)
                    step_bins = 20
                    num_examples = (len(generator_losses) // step_bins) * step_bins
                    plt.plot(
                        range(num_examples // step_bins), 
                        torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                        label="Generator Loss"
                    )
                    plt.plot(
                        range(num_examples // step_bins), 
                        torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss"
                    )
                    plt.legend()
                    plt.show()
            cur_step += 1
    torch.save(gen.state_dict(), '%s_%s.pt' % ('gan_gen', cur_step))
    torch.save(crit.state_dict(), '%s_%s.pt' % ('gan_disc', cur_step))
    

if __name__ == '__main__':
    main()
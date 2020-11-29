import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from TransformLayer import ColorJitterLayer
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import data as dataset
from generation_builder import ExperimentBuilder
import gan_generator

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

def gan_data(imgs, gen, z_dim = 64):
    device = imgs.device
    fake_noise = get_noise(imgs.shape[0], z_dim, device=device)
    fake = gen(fake_noise)
    fake = (fake+1)/2
    return fake

def random_crop(imgs, out=84):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped


def grayscale(imgs):
    # imgs: b x c x h x w
    device = imgs.device
    b, c, h, w = imgs.shape
    frames = c // 3
    
    imgs = imgs.view([b,frames,3,h,w])
    imgs = imgs[:, :, 0, ...] * 0.2989 + imgs[:, :, 1, ...] * 0.587 + imgs[:, :, 2, ...] * 0.114 
    
    imgs = imgs.type(torch.uint8).float()
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, :, None, :, :]
    imgs = imgs * torch.ones([1, 1, 3, 1, 1], dtype=imgs.dtype).float().to(device) # broadcast tiling
    return imgs

def random_grayscale(images,p=.3):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: cpu or cuda
        returns torch.tensor
    """
    device = images.device
    in_type = images.type()
    images = images * 255.
    images = images.type(torch.uint8)
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    images = images.to(device)
    gray_images = grayscale(images)
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] // 3
    images = images.view(*gray_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None, None]
    out = mask * gray_images + (1 - mask) * images
    out = out.view([bs, -1, h, w]).type(in_type) / 255.
    return out

# random cutout
# TODO: should mask this 

def random_cutout(imgs, min_cut=10,max_cut=30):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        min / max cut: int, min / max size of cutout 
        returns np.array
    """

    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        #print(img[:, h11:h11 + h11, w11:w11 + w11].shape)
        cutouts[i] = cut_img
    return cutouts

def random_cutout_color(imgs, min_cut=10,max_cut=30):
    """
        args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
    """
    
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        
        # add random box
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = np.tile(
            rand_box[i].reshape(-1,1,1),                                                
            (1,) + cut_img[:, h11:h11 + h11, w11:w11 + w11].shape[1:])
        
        cutouts[i] = cut_img
    return cutouts

# random flip

def random_flip(images,p=.2):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: cpu or gpu, 
        p: prob of applying aug,
        returns torch.tensor
    """
    # images: [B, C, H, W]
    device = images.device
    bs, channels, h, w = images.shape
    
    images = images.to(device)

    flipped_images = images.flip([3])
    
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] #// 3
    images = images.view(*flipped_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None]
    
    out = mask * flipped_images + (1 - mask) * images

    out = out.view([bs, -1, h, w])
    return out

# random rotation

def random_rotation(images,p=.3):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: str, cpu or gpu, 
        p: float, prob of applying aug,
        returns torch.tensor
    """
    device = images.device
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    
    images = images.to(device)

    rot90_images = images.rot90(1,[2,3])
    rot180_images = images.rot90(2,[2,3])
    rot270_images = images.rot90(3,[2,3])    
    
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    rnd_rot = np.random.randint(1, 4, size=(images.shape[0],))
    mask = rnd <= p
    mask = rnd_rot * mask
    mask = torch.from_numpy(mask).to(device)
    
    frames = images.shape[1]
    masks = [torch.zeros_like(mask) for _ in range(4)]
    for i,m in enumerate(masks):
        m[torch.where(mask==i)] = 1
        m = m[:, None] * torch.ones([1, frames]).type(mask.dtype).type(images.dtype).to(device)
        m = m[:,:,None,None]
        masks[i] = m
    
    
    out = masks[0] * images + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images

    out = out.view([bs, -1, h, w])
    return out


# random color

    

def random_convolution(imgs):
    '''
    random covolution in "network randomization"
    
    (imbs): B x (C x stack) x H x W, note: imgs should be normalized and torch tensor
    '''
    _device = imgs.device
    
    img_h, img_w = imgs.shape[2], imgs.shape[3]
    num_stack_channel = imgs.shape[1]
    num_batch = imgs.shape[0]
    num_trans = num_batch
    batch_size = int(num_batch / num_trans)
    
    # initialize random covolution
    rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)
    
    for trans_index in range(num_trans):
        torch.nn.init.xavier_normal_(rand_conv.weight.data)
        temp_imgs = imgs[trans_index*batch_size:(trans_index+1)*batch_size]
        temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
        rand_out = rand_conv(temp_imgs)
        if trans_index == 0:
            total_out = rand_out
        else:
            total_out = torch.cat((total_out, rand_out), 0)
    total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
    return total_out


def random_color_jitter(imgs):
    """
        inputs np array outputs tensor
    """
    b,c,h,w = imgs.shape
    imgs = imgs.view(-1,3,h,w)
    transform_module = nn.Sequential(ColorJitterLayer(brightness=0.4, 
                                                contrast=0.4,
                                                saturation=0.4, 
                                                hue=0.5, 
                                                p=1.0, 
                                                batch_size=128))

    imgs = transform_module(imgs).view(b,c,h,w)
    return imgs


def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs


def no_aug(x):
    return x

# RGB shift
def rgb_shift(imgs):
    """
        args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
    """
    red_shift = np.random.randint(-64, 64)
    green_shift = np.random.randint(-64, 64)
    blue_shift = np.random.randint(-64, 64)
    n, c, h, w = imgs.shape
    frames = c // 3
    imgs_copy = imgs.clone()
    imgs_copy = imgs_copy.view([n,frames,3,h,w])
    imgs_copy[:, :, 0, ...] += red_shift
    imgs_copy[:, :, 1, ...] += green_shift
    imgs_copy[:, :, 2, ...] += blue_shift
    imgs_copy = imgs_copy.reshape(n,-1,h,w)
    return imgs_copy

# Channel shuffle
def rgb_shuffle(imgs):
    n, c, h, w = imgs.shape
    frames = c // 3
    imgs_copy = imgs.clone()
    imgs_copy = imgs_copy.view([n,frames,3,h,w])
    red_frame = imgs_copy[:, :, 0, ...]
    green_frame = imgs_copy[:, :, 1, ...]
    blue_frame = imgs_copy[:, :, 2, ...]
    imgs_copy[:, :, 0, ...] = green_frame
    imgs_copy[:, :, 1, ...] = blue_frame
    imgs_copy[:, :, 2, ...] = green_frame
    imgs_copy = imgs_copy.reshape(n,-1,h,w)
    return imgs_copy

# Median Blur
def median_blur(imgs):
    n, c, h, w = imgs.shape
    frames = c // 3
    imgs_copy = imgs.copy()
    imgs_copy = imgs_copy.reshape([n,frames,3,h,w])
    for idx in range(n):
        for frame in range(frames):
            img_to_blur = imgs_copy[idx, frame, :, :, :]
            imgs_copy[idx, frame, :, :, :] = ndimage.median_filter(img_to_blur, 3)
    imgs_copy = imgs_copy.reshape(n,-1,h,w)
    return imgs_copy

# Image invert random
def img_invert(imgs):
    n, c, h, w = imgs.shape
    frames = c // 3
    imgs_invert = imgs.clone()
    imgs_invert = imgs_invert.view([n,frames,3,h,w])
    for idx in range(n):
        for frame in range(frames):
            rand = random.random()
            if rand>0.5:
                img_to_invert = imgs_invert[idx, frame, :, :, :]
                imgs_invert[idx, frame, :, :, :] = 255-img_to_invert
    imgs_invert = imgs_invert.reshape(n,-1,h,w)
    return imgs_invert

if __name__ == '__main__':
    import time 
    from tabulate import tabulate
    def now():
        return time.time()
    def secs(t):
        s = now() - t
        tot = round((1e5 * s)/60,1)
        return round(s,3),tot

    x = np.load('data_sample.npy',allow_pickle=True)
    x = np.concatenate([x,x,x],1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.from_numpy(x).to(device)
    x = x.float() / 255.

    #mix-up
    t = now()
    mixup_data(x.cpu().numpy())
    s1,tot1 = secs(t)
        # crop
    t = now()
    random_crop(x.cpu().numpy(),64)
    s1,tot1 = secs(t)
    # grayscale 
    t = now()
    random_grayscale(x,p=.5)
    s2,tot2 = secs(t)
    # normal cutout 
    t = now()
    random_cutout(x.cpu().numpy(),10,30)
    s3,tot3 = secs(t)
    # color cutout 
    t = now()
    random_cutout_color(x.cpu().numpy(),10,30)
    s4,tot4 = secs(t)
    # flip 
    t = now()
    random_flip(x,p=.5)
    s5,tot5 = secs(t)
    # rotate 
    t = now()
    random_rotation(x,p=.5)
    s6,tot6 = secs(t)
    # rand conv 
    t = now()
    random_convolution(x)
    s7,tot7 = secs(t)
    # rand color jitter 
    t = now()
    random_color_jitter(x)
    s8,tot8 = secs(t)
    # rgb shift
    t = now()
    rgb_shift(x)
    s9, tot9 = secs(t)
    # channel shuffle
    t = now()
    rgb_shuffle(x)
    s10, tot10 = secs(t)
    # Median blur
    t = now()
    median_blur(x_numpy)
    s11, tot11 = secs(t)
    # rand inversion
    t = now()
    img_invert(x)
    s12, tot12 = secs(t)

    print(tabulate([['Crop', s1,tot1], 
                    ['Grayscale', s2,tot2], 
                    ['Normal Cutout', s3,tot3], 
                    ['Color Cutout', s4,tot4], 
                    ['Flip', s5,tot5], 
                    ['Rotate', s6,tot6], 
                    ['Rand Conv', s7,tot7], 
                    ['Color Jitter', s8,tot8],
                    ['RGB Shift', s9, tot9],
                    ['Channel Shuffle', s10, tot10],
                    ['Median Blur', s11, tot11],
                    ['Random Inversion', s12, tot12],
                    ],
                    headers=['Data Aug', 'Time / batch (secs)', 'Time / 100k steps (mins)']))

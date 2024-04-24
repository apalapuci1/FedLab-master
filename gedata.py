from denoising.denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
from torch.utils.data import DataLoader, TensorDataset
data=[]
for i in range(45):
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 300,
        timesteps = 1000    # number of steps
    )
    training_data=torch.load('./datasets/new/class{}'.format(i))
    for epoch in range(20):
        training_images = training_data.unsqueeze(1) # images are normalized from 0 to 1
        loss = diffusion(training_images)
        loss.backward()
        # after a lot of training
        # sampled_images = diffusion.sample(batch_size = 4)
        # sampled_images.shape # (4, 3, 128, 128)
    for i in range(1000):
        sampled_images = diffusion.sample(batch_size=1)
        pass

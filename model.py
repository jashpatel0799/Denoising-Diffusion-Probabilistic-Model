import torch
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

UNET = UNet2DModel(sample_size = 64, in_channels = 3, out_channels = 3,
                   layers_per_block=2,  # how many ResNet layers to use per UNet block
                    # block_out_channels=(128, 256, 512, 512),  # the number of output channes for each UNet block
                    block_out_channels=(128, 256, 512, 512),  # the number of output channes for each UNet block

                    down_block_types=(
                    # a regular ResNet downsampling block
                        # "DownBlock2D",
                        # "DownBlock2D",
                        "DownBlock2D",
                        "DownBlock2D",
                        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                        "DownBlock2D",
                    ),
                    up_block_types=(
                        "UpBlock2D",  # a regular ResNet upsampling block
                        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                        "UpBlock2D",
                        "UpBlock2D",
                        # "UpBlock2D",
                        # "UpBlock2D"
                    )).to(device)

# for param in UNET.parameters():
    # param.requires_grad = True

Scheduler = DDPMScheduler(num_train_timesteps = 1001)
# load model and scheduler
DDPM = DDPMPipeline(UNET, Scheduler).to(device)
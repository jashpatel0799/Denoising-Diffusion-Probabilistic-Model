import torch
import einops
import numpy as np
from tqdm.auto import tqdm

def ddpm_train(ddpm, dataloader, loss_fn, optimizer, device):
    
    n_steps = 1000


    epoch_loss = 0.0
    for step, batch in enumerate(tqdm(dataloader)):
        # Load data
        x0 = batch[0].to(device)
        n = len(x0)

        # Picking some noise for each of the images in the batch
        eta = torch.randn_like(x0).to(device)
        t = torch.randint(0, n_steps, (n,)).to(device)

        # compating the noise base x0 and time stamp
        # noisy_imgs = ddpm(x0, t, eta)
        min_beta = ddpm.scheduler.config.beta_start
        max_beta = ddpm.scheduler.config.beta_end

        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alpha = 1 - betas
        alpha_bars = torch.tensor([torch.prod(alpha[:i + 1]) for i in range(len(alpha))]).to(device)
        a_bar = alpha_bars[t]
        noisy_imgs = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta

        # get mosdel estimate of noise based image  and  the time-step
        # eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

        # print(f"timestep shape: {t.shape}")
        # print(f"timestep Reshape: {t.reshape(n,-1).shape}")

        # print(f"timestep: {t}")
        # print(f"timestep reshape: {t.reshape(n, -1)}")
        eta_theta = ddpm.components['unet'](sample = noisy_imgs, timestep = t)

        # Optimizing the MSE between the noise plugged and predicted noise
        loss = loss_fn(eta_theta.sample, eta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() 

    epoch_loss /= len(dataloader)

    return epoch_loss, ddpm


def ddpm_test(ddpm, dataloader, loss_fn, device):
    n_steps = 1000

    epoch_loss = 0.0
    for step, batch in enumerate(tqdm(dataloader)):
        # Load data
        x0 = batch[0].to(device)
        n = len(x0)

        # Picking some noise for each of the images in the batch
        eta = torch.randn_like(x0).to(device)
        t = torch.randint(0, n_steps, (n,)).to(device)

        # compating the noise base x0 and time stamp
        # noisy_imgs = ddpm(x0, t, eta)
        min_beta = ddpm.scheduler.config.beta_start
        max_beta = ddpm.scheduler.config.beta_end

        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alpha = 1 - betas
        alpha_bars = torch.tensor([torch.prod(alpha[:i + 1]) for i in range(len(alpha))]).to(device)
        a_bar = alpha_bars[t]
        noisy_imgs = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta

        # get mosdel estimate of noise based image  and  the time-step
        # eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))
        eta_theta = ddpm.components['unet'](sample = noisy_imgs, timestep = t)

        # Optimizing the MSE between the noise plugged and predicted noise
        loss = loss_fn(eta_theta.sample, eta)
        

        epoch_loss += loss.item() 

    epoch_loss /= len(dataloader)

    return epoch_loss



def generate_new_image(ddpm, device, n_sample = 1, c=3, h = 64, w = 64):
    ddpm_n_steps = ddpm.scheduler.config.num_train_timesteps
    # frame_idxs = np.linspace(0, ddpm_n_steps, frames_per_gif).astype(np.uint)
    # frames = []

    with torch.no_grad():

        x = torch.randn(n_sample, c, h, w).to(device)
        for idx, t in enumerate(list(range(ddpm_n_steps))[::-1]):
            time_tensor = (torch.ones(n_sample, 1) * t).squeeze(dim=0).to(device).long()
            # print()
            # print(time_tensor)
            # print(time_tensor.shape)
            # print(time_tensor.squeeze())
            # print(time_tensor.squeeze().shape)
            # print()
        #   eta_theta = ddpm.backward(x, time_tensor)
            eta_theta = ddpm.components['unet'](sample = x, timestep = time_tensor.squeeze())
            # t = torch.randint(0, ddpm_n_steps, (n_sample,)).to(device)

            # compating the noise base x0 and time stamp
            # noisy_imgs = ddpm(x0, t, eta)
            min_beta = ddpm.scheduler.config.beta_start
            max_beta = ddpm.scheduler.config.beta_end

            betas = torch.linspace(min_beta, max_beta, ddpm_n_steps).to(device)
            alpha = 1 - betas
            alpha_bars = torch.tensor([torch.prod(alpha[:i + 1]) for i in range(len(alpha))]).to(device)
            # a_bar = alpha_bars[t]
            # noisy_imgs = a_bar.sqrt().reshape(n_sample, 1, 1, 1) * x + (1 - a_bar).sqrt().reshape(n_sample, 1, 1, 1) * eta

            # get mosdel estimate of noise based image  and  the time-step
            # eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))
            # ---------------- 

            alpha_t = alpha[t]
            alpha_t_bar = alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta.sample)

            if t > 0:
                z = torch.randn(n_sample, c, h, w).to(device)

                beta_t = betas[t]
                sigma_t = beta_t.sqrt()

                x = x + sigma_t * z

            # if idx in frame_idxs or t == 0:
            #     normalized = x.clone()

            #     for i in range(len(normalized)):
            #         normalized[i] -= torch.min(normalized[i])
            #         normalized[i] *= 255 / torch.max(normalized[i])

            #         frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_sample ** 0.5))
            #         frame = frame.cpu().numpy().astype(np.uint8)

            #         frames.append(frame)
    # print(x.shape)
    return x
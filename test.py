import os
import torch
import torch.nn as nn
from model import DDPM
from data import lfw_test_dataloader
from torchmetrics.image.fid import FrechetInceptionDistance
from engine import ddpm_train, ddpm_test, generate_new_image



fid = FrechetInceptionDistance(feature=192)

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

loss_fn = nn.MSELoss()


last_epoch = 300
PATH = f"check_point/LFW_checkpoint_{last_epoch}.pth"
isExist = os.path.exists(PATH)

optimizer = torch.optim.Adam(DDPM.components['unet'].parameters(), lr = 0.001)

if isExist:
    check_point = torch.load(PATH)
    DDPM.components['unet'].load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    epoch = check_point['epoch'] + 1
    loss = check_point['loss']

# print(isExist)
# print(optimizer)
# print(epoch)
# act_images = [image for batch, (image, _) in enumerate(lfw_test_dataloader)]
act_images = []
i = 0
for batch, (image, _) in enumerate(lfw_test_dataloader):
    for im in image:
        if i == 100:
            break
        act_images.append(im)
        i += 1
    
act_images = torch.stack(act_images).type(torch.uint8)
# print(act_images.shape)
# print(act_images)
# print(act_images[0].dtype, act_images[0])
# print(type(act_images[0]))
# for i in act_images:
#     print(type(i), i)
#     break
gen_images = generate_new_image(DDPM, device, n_sample = 100).to(torch.uint8)
# gen_images = 

fid.update(act_images, real=True)
fid.update(gen_images.to('cpu'), real=False)
fid_val = fid.compute()


print(fid_val.item())
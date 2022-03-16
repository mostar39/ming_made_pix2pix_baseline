import numpy as np
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from lpips import lpips
from Dataset import ImageDataset
from torch.utils.data import DataLoader

transforms_ = transforms.Compose([
    transforms.Resize((256,256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

prediction_root = 'side_test_data/vgg_aug_test_data'
len_pred = len(os.listdir(prediction_root))
prediction_image = [os.path.join(prediction_root,str(i)+'.jpg') for i in range(len_pred)]

GT_path = [os.path.join('side_images/after/val',str(i+1)+'.jpg') for i in range(len(os.listdir('side_images/after/val')))]
AUG_root = '/home/ylab3/improved_CcGAN/365_data/RC-49_256x256/CcGAN-improved/Data_Augmentation_14000/after/'
AUG_name = os.listdir(AUG_root)
AUG_name.sort()
AUG_path= [os.path.join(AUG_root,AUG_name[i]) for i in range(len(AUG_name))]
AUG_path = AUG_path[1800:-12]
All_GT = GT_path + AUG_path

All_GT = All_GT[:37]
prediction_image = prediction_image[:37]

train_dataset = ImageDataset(prediction_image,All_GT,transforms_)
train_dataloader = DataLoader(train_dataset, batch_size=25, num_workers=4)


loss_fn_alex = lpips.LPIPS('alex')
loss_fn_alex.cuda()
loss_list = []

for i, batch in enumerate(train_dataloader) :
    loss=loss_fn_alex.forward(batch['before'].cuda(), batch['after'].cuda())
    loss_list.append(loss.item())

print(np.mean(loss_list))

# loss_fn_alex = lpips.LPIPS('vgg')
# loss_fn_alex.cuda()
# loss_lpips_alex = loss_fn_alex.forward(real_after,fake_after)

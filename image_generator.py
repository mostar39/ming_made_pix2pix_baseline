import torch
import os
from Model import GeneratorUNet
from Dataset import ImageDataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np

generator = GeneratorUNet()
L1_aug_best = torch.load('vgg_aug/best_model.pth')
generator.load_state_dict(L1_aug_best)
generator = generator.cuda()
generator.eval()

real_test_image = ['side_images/before/val/'+str(i+1)+'.jpg' for i in range(37)]

aug_before_root = '/home/ylab3/improved_CcGAN/365_data/RC-49_256x256/CcGAN-improved/Data_Augmentation_14000/before/'
aug_before_name = os.listdir(aug_before_root)
aug_before_name.sort()

aug_before_list = []
for i in range(len(aug_before_name)) :
    aug_before_list.append(os.path.join(aug_before_root,aug_before_name[i]))

all_test_root = real_test_image + aug_before_list[1800:-12]


transforms_ = transforms.Compose([
    transforms.Resize((256,256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_dataset = ImageDataset(all_test_root,all_test_root,transforms_)
test_dataloader = DataLoader(test_dataset, batch_size=25, num_workers=4)


os.makedirs('vgg_aug_test_data', exist_ok=True)

fake_all_image= []

for i, batch in enumerate(test_dataloader) :
    real_before = batch['before'].cuda()
    fake_after = generator(real_before)
    fake_after = fake_after*0.5+0.5
    fake_after = fake_after*255.0
    fake_after = fake_after.type(torch.uint8)
    fake_all_image.append(fake_after.cpu())

fake_all_image = torch.cat(fake_all_image,dim=0)
fake_all_image = fake_all_image.numpy()


for k in range(len(fake_all_image)) :
    file_name = os.path.join('vgg_aug_test_data',str(k)+'.jpg')
    image_i = fake_all_image[k].astype(np.uint8)
    image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
    image_i_pil.save(file_name)



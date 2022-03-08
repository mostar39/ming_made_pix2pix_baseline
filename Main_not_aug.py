import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import preprocessing

import torch
import pandas as pd
import torch.nn as nn
from copy import deepcopy

from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from Model import GeneratorUNet, Discriminator_1
from Dataset import ImageDataset
import pytorch_ssim
from perceptual_loss import VGGPerceptualLoss
import pickle

transforms_ = transforms.Compose([
    transforms.Resize((256,256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

#------------------------DataRoot..-----------------------
before_train_root = os.listdir('pair_data/before/train')
before_test_root = os.listdir('pair_data/before/val')
after_train_root = os.listdir('pair_data/after/train')
after_test_root = os.listdir('pair_data/after/val')

before_number_train =[]
before_number_test = []
after_number_train = []
after_number_test = []

for i in range(len(before_train_root)) :
    before_number_train.append(before_train_root[i][:-4])
    after_number_train.append(after_train_root[i][:-4])

for i in range(len(after_test_root)) :
    before_number_test.append(before_test_root[i][:-4])
    after_number_test.append(after_test_root[i][:-4])

before_train_df = pd.DataFrame()
before_train_df['number'] = before_number_train
before_train_df['number'] = before_train_df['number'].astype(int)
before_train_df['file_name'] = before_train_root
before_train_df= before_train_df.sort_values(by=['number'], axis=0)

before_test_df = pd.DataFrame()
before_test_df['number'] = before_number_test
before_test_df['number'] = before_test_df['number'].astype(int)
before_test_df['file_name'] = before_test_root
before_test_df= before_test_df.sort_values(by=['number'], axis=0)

train_data_numbering = before_train_df['file_name']
test_data_numbering = before_test_df['file_name']

train_before_root = list('side_images/before/train/' + train_data_numbering)
train_after_root = list('side_images/after/train/' + train_data_numbering)
test_before_root = list('side_images/before/val/' + test_data_numbering)
test_after_root = list('side_images/after/val/' + test_data_numbering)

# aug_before_root = '/home/ylab3/improved_CcGAN/365_data/RC-49_256x256/CcGAN-improved/Data_Augmentation_14000/before/'
# aug_before_name = os.listdir(aug_before_root)
# aug_before_name.sort()
#
#
# aug_after_root = '/home/ylab3/improved_CcGAN/365_data/RC-49_256x256/CcGAN-improved/Data_Augmentation_14000/after/'
# aug_after_name = os.listdir(aug_after_root)
# aug_after_name.sort()
#
# aug_before_list = []
# aug_after_list = []
# for i in range(len(aug_before_name)) :
#     aug_before_list.append(os.path.join(aug_before_root,aug_before_name[i]))
#     aug_after_list.append(os.path.join(aug_after_root, aug_after_name[i]))
#
# train_before_root = train_before_root + aug_before_list[:1800]
# train_after_root = train_after_root + aug_after_list[:1800]
# test_before_root = test_before_root + aug_before_list[1800:]
# test_after_root = test_after_root + aug_after_list[1800:]

#-----------------------------------------------------------------
train_dataset = ImageDataset(train_before_root,train_after_root,transforms_)
test_dataset = ImageDataset(test_before_root,test_after_root,transforms_)

#file_name : train_dataloader.dataset.before
train_dataloader = DataLoader(train_dataset, batch_size=20, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=20, num_workers=4)
#-----------------------------------------------------------------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

generator = GeneratorUNet()
discriminator1 = Discriminator_1()

generator.cuda()
discriminator1.cuda()

generator.apply(weights_init_normal)
discriminator1.apply(weights_init_normal)

criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

criterion_GAN.cuda()
criterion_pixelwise.cuda()

perceptual=VGGPerceptualLoss()
perceptual.cuda()

# def get_scheduler(optimizer) :
#     def lambda_rule(epoch) :
#         #epoch_count = 1
#         #n_epoch = 2000 -> epoch
#         #n_epochs_decay = 100
#         lr_l = 1.0 - max(0, epoch + 1 - 1800) / float(100 + 1)
#         return lr_l
#
#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
#     return scheduler


lr = 0.0002

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5,0.999))
optimizer_D_1 = torch.optim.Adam(discriminator1.parameters(), lr=lr, betas=(0.5,0.999))

import time
n_epoch = 2000
sample_interval = 200
lambda_pixel = 100

gan_loss = []
D_loss = []
pixel_loss = []

test_gan_loss = []
test_D_loss = []
test_pixel_loss = []

best_model = None
best_model_score = 0


for epoch in range(n_epoch) :
    start_time = time.time()
    for i, batch in enumerate(train_dataloader) :
        generator.train()
        discriminator1.train()
        #batch.keys() : before, after, target_bmi
        real_before = batch['before'].cuda()
        real_after = batch['after'].cuda()


        real_dis1 = torch.cuda.FloatTensor(real_before.size(0),1, 16, 16).fill_(1.0)
        fake_dis1 = torch.cuda.FloatTensor(real_before.size(0),1, 16, 16).fill_(0.0)

        optimizer_G.zero_grad()

        # def forward(self, x, bmi)
        fake_after = generator(real_before)
        #Generator Loss
        # def forward(self, img_A, img_B): Discriminator1
        # def forward(self,x_image, bmi): Discriminator2
        loss_GAN_to_dis1 = criterion_GAN(discriminator1(real_before, fake_after), real_dis1)
        # loss_GAN_to_dis2 = criterion_GAN(discriminator2(fake_after, target_bmi), real_dis2)
        loss_pixel = criterion_pixelwise(real_after, fake_after)
        # for_ssim = torch.cuda.FloatTensor(1,1).fill_(1.0)
        # loss_ssim=pytorch_ssim.ssim(real_after,fake_after)
        # real_ssim_loss = for_ssim - loss_ssim
        loss_vgg = perceptual(real_after, fake_after)

        loss_G = loss_GAN_to_dis1 + lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()


        #-----------------Discriminator_1 & Discriminator_2-----------------
        optimizer_D_1.zero_grad()

        D1_loss_real = criterion_GAN(discriminator1(real_before,real_after),real_dis1)
        D1_loss_fake = criterion_GAN(discriminator1(fake_after.detach(), real_after), fake_dis1)
        loss_D1 = (D1_loss_real + D1_loss_fake) / 2

        loss_D1.backward()
        optimizer_D_1.step()

        # optimizer_D_2.zero_grad()
        #
        # D2_loss_real = criterion_GAN(discriminator2(real_after,target_bmi),real_dis2)
        # D2_loss_fake = criterion_GAN(discriminator2(fake_after.detach(), target_bmi), fake_dis2)
        # loss_D2 = (D2_loss_real + D2_loss_fake) / 2

        # loss_D2.backward()
        # optimizer_D_2.step()

        #-------------------------------------------------------------------

        done = epoch * len(train_dataloader) + i
        if done % sample_interval == 0 :
            generator.eval()
            discriminator1.eval()

            val_data = next(iter(test_dataloader))
            real_before_test = val_data['before'].cuda()
            real_after_test = val_data['after'].cuda()

            fake_after_test = generator(real_before_test)

            gan_loss.append(loss_G.item())
            pixel_loss.append(loss_pixel.item())
            D_loss.append(loss_D1.item())

            # test_gan_loss = []
            # test_D_loss = []
            # test_pixel_loss = []

            loss_GAN_to_dis1_test = criterion_GAN(discriminator1(real_before_test, fake_after_test), real_dis1)
            loss_pixel_test = criterion_pixelwise(real_after_test, fake_after_test)
            loss_vgg_test = perceptual(real_after_test, fake_after_test)

            loss_G_test = loss_GAN_to_dis1_test + lambda_pixel * loss_pixel_test

            D1_loss_real_test = criterion_GAN(discriminator1(real_before_test, real_after_test), real_dis1)
            D1_loss_fake_test = criterion_GAN(discriminator1(fake_after_test.detach(), real_after_test), fake_dis1)
            loss_D1_test = (D1_loss_real_test + D1_loss_fake_test) / 2

            test_gan_loss.append(loss_G_test.item())
            test_D_loss.append(loss_D1_test.item())
            test_pixel_loss.append(loss_pixel_test.item())

            if (loss_G_test.item() < best_model_score) and (epoch >= 400) :
                best_model_score = loss_G_test.item()
                best_model = deepcopy(generator.state_dict())

            img_sample = torch.cat((real_before_test.data, fake_after_test.data, real_after_test.data),-2)
            save_image(img_sample,f"L1_side_not_aug/{done}.png", nrow=5, normalize=True)


    print(f"[Epoch {epoch}/{n_epoch}] [D1 loss: {loss_D1.item():.6f}][G pixel loss: {loss_pixel.item():.6f},"
          f" adv loss: {loss_G.item()}] [Elapsed time: {time.time() - start_time:.2f}s]")

torch.save(best_model,'L1_side_not_aug/best_model.pth')

import matplotlib.pyplot as plt
x = [i*200 for i in range(len(gan_loss))]

plt.figure(figsize=(30,10))
plt.subplot(1,3,1)
plt.plot(x, gan_loss, label = 'Generator loss_train', color = 'royalblue')
plt.plot(x, test_gan_loss, label = 'Generator loss_test')
plt.legend(loc=0)
plt.title('Generator loss', fontsize=20)
plt.subplot(1,3,2)
plt.plot(x, D_loss, label = 'Discriminator loss', color = 'mediumpurple')
plt.plot(x, test_D_loss, label = 'Discriminator loss')
plt.legend(loc=0)
plt.title('Discriminator loss', fontsize=20)
plt.subplot(1,3,3)
plt.plot(x, pixel_loss, label = 'L1 loss', color = 'plum')
plt.plot(x, test_pixel_loss, label = 'L1 loss')
plt.legend(loc=0)
plt.title('L1 loss', fontsize=20)

plt.savefig('L1_side_not_aug/loss_fig.jpg')


csv_file = pd.DataFrame()
csv_file['gan_loss'] = gan_loss
csv_file['gan_loss_test'] = test_gan_loss
csv_file['d_loss'] = D_loss
csv_file['test_D_loss'] = test_D_loss
csv_file['pixel_loss'] = pixel_loss
csv_file['test_pixel_loss'] = test_pixel_loss

csv_file.to_excel('L1_side_not_aug/loss_excel.xlsx')


import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

#
# import pytorch_ssim
# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
#
# transforms_ = transforms.Compose([
#     transforms.Resize((224,224), Image.BICUBIC),
#     transforms.ToTensor(),
# ])
#
# side_per_loss = []
#
# perceptual=VGGPerceptualLoss()
#
# for i in range(30) :
#
#     image_real = Image.open('/home/ylab3/pytorch-CycleGAN-and-pix2pix/results/facades_pix2pix/test_latest/images/'+str(i+1)+'_real_B.png')
#     image_fake = Image.open('/home/ylab3/pytorch-CycleGAN-and-pix2pix/results/facades_pix2pix/test_latest/images/'+str(i+1)+'_fake_B.png')
#
#     image_real = image_real.convert('L')
#     image_fake = image_fake.convert('L')
#
#     image_real = transforms_(image_real)
#     image_fake = transforms_(image_fake)
#
#
#     image_real = torch.reshape(image_real, (1, 1, 224, 224))
#     image_fake = torch.reshape(image_fake, (1, 1, 224, 224))
#
#     side_per_loss.append(perceptual(image_real,image_fake).item())
#
# np.mean(side_per_loss)

# front, black and white -> mean : 2.0728, max : 2.84, min : 1.6373
# side -> mean : 2.14 / max : 2.60 / min : 1.9143
# side black and white -> mean : 1.800 / max : 2.19 / min : 1.577
import torch
import torch.nn as nn

# U-Net 아키텍처의 다운 샘플링(Down Sampling) 모듈
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        # 너비와 높이가 2배씩 감소
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# U-Net 아키텍처의 업 샘플링(Up Sampling) 모듈: Skip Connection 사용
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        # 너비와 높이가 2배씩 증가
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1) # [batchsize, channel, 3, 3]

        return x

# class MLP_BMI(nn.Module) :
#     def __init__(self):
#         super(MLP_BMI, self).__init__()
#         layers = [nn.Linear(513,256)]
#         layers.append(nn.Dropout(0.2))
#         layers.append(nn.Linear(256,128))
#         layers.append(nn.LeakyReLU(0.2))
#         layers.append(nn.Dropout(0.2))
#         layers.append(nn.LeakyReLU(0.2))
#         layers.append(nn.Linear(128,256))
#         layers.append(nn.Dropout(0.2))
#         layers.append(nn.LeakyReLU(0.2))
#         layers.append(nn.Linear(256,512))
#         layers.append(nn.LeakyReLU(0.2))
#
#         self.model = nn.Sequential(*layers)
#
#     def forward(self,x):
#
#         x = self.model(x)
#
#         return x


# U-Net 생성자(Generator) 아키텍처
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False) # 출력: [64 X 128 X 128]
        self.down2 = UNetDown(64, 128) # 출력: [128 X 64 X 64]
        self.down3 = UNetDown(128, 256) # 출력: [256 X 32 X 32]
        self.down4 = UNetDown(256, 512, dropout=0.5) # 출력: [512 X 16 X 16]
        self.down5 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 8 X 8]
        self.down6 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 4 X 4]
        self.down7 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 2 X 2]
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5) # 출력: [512 X 1 X 1]

        # Skip Connection 사용(출력 채널의 크기 X 2 == 다음 입력 채널의 크기)
        self.up1 = UNetUp(512, 512, dropout=0.5) # 출력: [1024 X 2 X 2]
        self.up2 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 4 X 4]
        self.up3 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 8 X 8]
        self.up4 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 16 X 16]
        self.up5 = UNetUp(1024, 256) # 출력: [512 X 32 X 32]
        self.up6 = UNetUp(512, 128) # 출력: [256 X 64 X 64]
        self.up7 = UNetUp(256, 64) # 출력: [128 X 128 X 128]

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), # 출력: [128 X 256 X 256]
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1), # 출력: [3 X 256 X 256]
            nn.Tanh(),
        )

        # self.mlp = MLP_BMI()

    def forward(self, x):
        # 인코더부터 디코더까지 순전파하는 U-Net 생성자(Generator)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # #welcome MLP
        # resize_d7=d8.reshape(-1,512)
        # concat=torch.cat([resize_d7,bmi],dim=1)
        # result_MLP=self.mlp(concat)
        # plus_latent= result_MLP.reshape(-1,512,1,1)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


# U-Net 판별자(Discriminator) 아키텍처
class Discriminator_1(nn.Module): #paired before and after
    def __init__(self, in_channels=3):
        super(Discriminator_1, self).__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            # 너비와 높이가 2배씩 감소
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # 두 개의 이미지(실제/변환된 이미지, 조건 이미지)를 입력 받으므로 입력 채널의 크기는 2배
            *discriminator_block(in_channels * 2, 64, normalization=False), # 출력: [64 X 128 X 128]
            *discriminator_block(64, 128), # 출력: [128 X 64 X 64]
            *discriminator_block(128, 256), # 출력: [256 X 32 X 32]
            *discriminator_block(256, 512), # 출력: [512 X 16 X 16]
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False) # 출력: [1 X 16 X 16]
        )

    # img_A: 실제/변환된 이미지, img_B: 조건(condition)
    def forward(self, img_A, img_B):
        # 이미지 두 개를 채널 레벨에서 연결하여(concatenate) 입력 데이터 생성
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
#
# class Discriminator_2_block(nn.Module) :
#     def __init__(self, in_channels, out_channels):
#         super(Discriminator_2_block,self).__init__()
#
#         layers = [nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=2)]
#         layers.append(nn.LeakyReLU(0.2))
#
#         self.model = nn.Sequential(*layers)
#
#     def forward(self,x):
#         return self.model(x)
#
#
# class Discriminator_2(nn.Module):
#     def __init__(self, in_channels=4):
#         # 3 : image, 1 : bmi
#         super(Discriminator_2, self).__init__()
#
#         self.dis_1 = Discriminator_2_block(in_channels,64)
#         self.dis_2 = Discriminator_2_block(64,128)
#         self.dis_3 = Discriminator_2_block(128,128)
#         self.last = nn.Linear(128,1)
#
#     def forward(self,x_image, bmi):
#         bmi=torch.unsqueeze(bmi,1)
#         bmi = torch.unsqueeze(bmi, 1)
#         # print(bmi.size())
#         bmi_re=bmi.repeat((1,1,256,256))
#         concat = torch.cat([x_image,bmi_re],dim=1)
#         output = self.dis_1(concat)
#         output = self.dis_2(output)
#         output = self.dis_3(output)
#         output = nn.AdaptiveMaxPool2d(output_size=1)(output)
#         output = torch.squeeze(output)
#         output = self.last(output)
#
#
#         return output



# bmi_test=torch.randn((5,1))
# x_image = torch.randn((5,3,256,256))
#
# dis2_test=Discriminator_2()
# a=dis2_test(x_image,bmi_test)
#
#
# bmi=torch.unsqueeze(bmi_test,1) #(5,1,1)
# bmi = torch.unsqueeze(bmi, 1) #(5,1,1,1)
# bmi_re = bmi.repeat((1, 1, 256, 256)) #(5,1,256,256)
# concat = torch.cat([x_image, bmi_re], dim=1) #(5,4,256,256)
#


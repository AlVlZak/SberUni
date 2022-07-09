import torch
import torch.nn as nn
import torch.nn.functional as F


def unet_subpart(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )
    return conv#.cuda()


def unet_transpose(in_c, out_c):
    conv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=2,
            stride=2
    )
    return conv#.cuda()


class SegmenterModel(nn.Module):
    def __init__(self):
        super(SegmenterModel, self).__init__()
        #self.init_ch = 64 # число каналов после первой свёртки
        #self.n_levels = 3 # число уровней до "основания" параболы
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drpout = nn.Dropout(0.1)
        
        self.in_ = unet_subpart(3, 64)
        
        self.down1 = unet_subpart(64, 128)
        self.down2 = unet_subpart(128, 256)
        self.down3 = unet_subpart(256, 512)
        self.down4 = unet_subpart(512, 1024)

        self.up_trans_1 = unet_transpose(1024, 512)
        self.up1 = unet_subpart(1024, 512)

        self.up_trans_2 = unet_transpose(512, 256)
        self.up2 = unet_subpart(512, 256)

        self.up_trans_3 = unet_transpose(256, 128)
        self.up3 = unet_subpart(256, 128)

        self.up_trans_4 = unet_transpose(128, 64)
        self.up4 = unet_subpart(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, image):
        x1 = self.in_(image)
        
        x2 = self.max_pool_2x2(x1)
        x2 = self.drpout(x2)
        
        x3 = self.down1(x2)
        
        x4 = self.max_pool_2x2(x3)
        x4 = self.drpout(x4)
        
        x5 = self.down2(x4)
        
        x6 = self.max_pool_2x2(x5)
        x6 = self.drpout(x6)
        
        x7 = self.down3(x6)
        
        x8 = self.max_pool_2x2(x7)
        x8 = self.drpout(x8)

        x9 = self.down4(x8)

        x = self.up_trans_1(x9)
        x = self.up1(torch.cat([x7, x], 1))

        x = self.up_trans_2(x)
        x = self.up2(torch.cat([x5, x], 1))

        x = self.up_trans_3(x)
        x = self.up3(torch.cat([x3, x], 1))

        x = self.up_trans_4(x)
        x = self.up4(torch.cat([x1, x], 1))

        x = self.out(x)
        return x#.cuda()
    
    def predict(self, x):
        # на вход подаётся одна картинка, а не батч, поэтому так
        y = self.forward(x.unsqueeze(0).cuda())
        return (y > 0).squeeze(0).squeeze(0).float().cuda()

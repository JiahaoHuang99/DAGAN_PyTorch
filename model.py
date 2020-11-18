import torch
import torch.nn as nn
import torchvision.models as models


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # set parameter
        self.df_dim = 64
        self.fin = 8192

        # network
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.df_dim, kernel_size=5, stride=2, padding=2, ),
            nn.LeakyReLU(negative_slope=0.2)

        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim, out_channels=self.df_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 2, out_channels=self.df_dim * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=self.df_dim * 4),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 4, out_channels=self.df_dim * 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=self.df_dim * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 8, out_channels=self.df_dim * 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=self.df_dim * 16),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 16, out_channels=self.df_dim * 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=self.df_dim * 32),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 32, out_channels=self.df_dim * 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.df_dim * 16),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 16, out_channels=self.df_dim * 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.df_dim * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.res8 = nn.Sequential(
            nn.Conv2d(in_channels=self.df_dim * 8, out_channels=self.df_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=self.df_dim * 2, out_channels=self.df_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=self.df_dim * 2, out_channels=self.df_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.df_dim * 8),
        )

        self.LRelu = nn.LeakyReLU(negative_slope=0.2)

        self.out = nn.Sequential(
            nn.Linear(self.fin, 1),
            nn.Sigmoid()
        )

    # forward propagation
    def forward(self, input_image):
        net_in = input_image
        net_h0 = self.conv0(net_in)
        net_h1 = self.conv1(net_h0)
        net_h2 = self.conv2(net_h1)
        net_h3 = self.conv3(net_h2)
        net_h4 = self.conv4(net_h3)
        net_h5 = self.conv5(net_h4)
        net_h6 = self.conv6(net_h5)
        net_h7 = self.conv7(net_h6)
        res_h7 = self.res8(net_h7)
        net_h8 = self.LRelu(res_h7 + net_h7)
        net_ho = net_h8.contiguous().view(net_h8.size(0), -1)
        logits = self.out(net_ho)

        return logits


# U_Net
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # set parameter
        self.gf_dim = 64
        self.kernel_size = 4
        self.padding = 1

        # network
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.gf_dim, kernel_size=self.kernel_size, stride=2,
                      padding=self.padding),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim, out_channels=self.gf_dim * 2, kernel_size=self.kernel_size, stride=2,
                      padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 2),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim * 2, out_channels=self.gf_dim * 4, kernel_size=self.kernel_size, stride=2,
                      padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 4),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim * 4, out_channels=self.gf_dim * 8, kernel_size=self.kernel_size, stride=2,
                      padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim * 8, out_channels=self.gf_dim * 8, kernel_size=self.kernel_size, stride=2,
                      padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim * 8, out_channels=self.gf_dim * 8, kernel_size=self.kernel_size, stride=2,
                      padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim * 8, out_channels=self.gf_dim * 8, kernel_size=self.kernel_size, stride=2,
                      padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim * 8, out_channels=self.gf_dim * 8, kernel_size=self.kernel_size, stride=2,
                      padding=self.padding),
            nn.LeakyReLU(negative_slope=0.2)  # 源代码又LReLu 论文框架没有
        )

        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim * 8, out_channels=self.gf_dim * 8, kernel_size=self.kernel_size,
                               stride=2, padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 8),
            nn.ReLU()
        )

        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim * 16, out_channels=self.gf_dim * 16,
                               kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 16),
            nn.ReLU()
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim * 24, out_channels=self.gf_dim * 16,
                               kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 16),
            nn.ReLU()
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim * 24, out_channels=self.gf_dim * 16,
                               kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 16),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim * 24, out_channels=self.gf_dim * 4, kernel_size=self.kernel_size,
                               stride=2, padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 4),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim * 8, out_channels=self.gf_dim * 2, kernel_size=self.kernel_size,
                               stride=2, padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim * 2),
            nn.ReLU()
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim * 4, out_channels=self.gf_dim, kernel_size=self.kernel_size,
                               stride=2, padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim),
            nn.ReLU()
        )

        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.gf_dim * 2, out_channels=self.gf_dim, kernel_size=self.kernel_size,
                               stride=2, padding=self.padding),
            nn.BatchNorm2d(num_features=self.gf_dim),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

        self.refine = nn.Tanh()

    # forward propagation
    def forward(self, x, is_refine=False):
        input = x
        down1 = self.conv1(input)
        down2 = self.conv2(down1)
        down3 = self.conv3(down2)
        down4 = self.conv4(down3)
        down5 = self.conv5(down4)
        down6 = self.conv6(down5)
        down7 = self.conv7(down6)
        down8 = self.conv8(down7)
        up7 = self.deconv7(down8)
        up6 = self.deconv6(torch.cat((down7, up7), 1))
        up5 = self.deconv5(torch.cat((down6, up6), 1))
        up4 = self.deconv4(torch.cat((down5, up5), 1))
        up3 = self.deconv3(torch.cat((down4, up4), 1))
        up2 = self.deconv2(torch.cat((down3, up3), 1))
        up1 = self.deconv1(torch.cat((down2, up2), 1))
        up0 = self.deconv0(torch.cat((down1, up1), 1))
        output = self.out(up0)

        if is_refine:
            output = self.refine(output + input)

        return output


class VGG_CNN(nn.Module):
    def __init__(self):
        super(VGG_CNN, self).__init__()

        # load vgg16
        self.vgg16 = models.vgg16(pretrained=True)
        # get conv4
        self.vgg16_cnn = nn.Sequential(*list(self.vgg16.features.children())[:-7])

    def forward(self, x):
        return self.vgg16_cnn(x)


# main
if __name__ == "__main__":
    pass

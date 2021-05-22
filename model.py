import torch.nn as nn

class DCGAN(nn.Module):
    def __init__(self, nz, ngf, ndf, nch):
        super(DCGAN, self).__init__()
        # Dimension of latent vector z.
        self.nz = nz
        # Dimension of generator feature map.
        self.ngf = ngf
        # Dimension of discriminator feature map.
        self.ndf = ndf
        # Dimension of generator output image channel.
        self.nch = nch

        # Generator
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, nch, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Conv2d(nch, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        # Initialize the weights.
        self.generator.apply(self._init_weights)
        self.discriminator.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """ Initialize the weights of Convolution and Batch Normalization layers with normal distribution.
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def dis_run(self, x):
        # Run the discriminator
        # Input: Image tensor x (b, nch, h, w)
        return self.discriminator(x)

    def gen_run(self, z):
        # Run the generator
        # Input: latent vector z (b, nz)
        return self.generator(z)
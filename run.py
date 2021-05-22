import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from model import DCGAN
from dataset import CelebA

class GANTrainer():
    def __init__(self, id, bone, real_dataloader, optimizer_gen, optimizer_dis, iterations=100000, n_sample=64, log_interval=50, save_interval=500):
        # Run id.
        self.id = id

        # A nn.Module containing GAN
        # Essential objects:
        #   - bone.generator: Input latent vector z, output fake image x
        #   - bone.discriminator: Input image tensor x, output real/fake score
        self.bone = bone
        # The device to use in the whole training procedure.
        self.device = next(self.bone.parameters()).device
        
        # Dataloader of real images.
        self.real_dataloader = real_dataloader

        # Optimizer of generator and discriminator. (Always use Adams instead of SGD)
        self.optimizer_gen = optimizer_gen
        self.optimizer_dis = optimizer_dis

        # Total iterations to run.
        self.iterations = iterations
        
        # Real/fake labels of the output of discriminator.
        self.fake_label = 0.
        self.real_label = 1.

        # Intervals of logging and saving checkpoint/sample images.
        self.log_interval = log_interval
        self.save_interval = save_interval

        # A fixed latent vector z. Random. Used as generating the sample output image.
        self.sample_z = torch.randn(n_sample, self.bone.nz, 1, 1).to(self.device)

        self.dist_path = os.path.join("dist", str(id))
        if not os.path.exists(self.dist_path):
            os.makedirs(self.dist_path)

    @staticmethod
    def freeze(model):
        """ Freeze the parameters of the specific model.
        """
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze(model):
        """ Unfreeze the parameters of the specific model.
        """
        for param in model.parameters():
            param.requires_grad = True

    def criterion(self, output, target):
        """ The computation of loss.
        """
        loss = nn.BCELoss()
        return loss(output, target)

    def train(self):
        """ Training procedure.
        Part I. Real image on discriminator
        1. Real image forward through discriminator.
        2. Real loss backward through discriminator, get real gradients.
        
        Part II. Fake image on discriminator
        3. Latent vector forward through generator, get fake image.
        4. Fake image forward through discriminator.
        5. Fake loss backward through discriminator, get fake gradients, accumulate on discriminator parameters.
        6. Discriminator parameters update via optimizer.
        
        Part III. Fake image on generator
        7. Fake image forward through discriminator(updated).
        8. Fake loss backward through discriminator and generator, with discriminator untrack of gradients. Get generator gradients.
        9. Generator parameters update via optimizer.
        """
        pbar = tqdm(total=self.iterations, leave=False)
        
        # Current iteration index.
        cur_iter = 0
        while True:
            for batch_idx, (data, _) in enumerate(self.real_dataloader):
                b, c, h, w = data.shape
                self.bone.generator.zero_grad()
                self.bone.discriminator.zero_grad()

                # Real image on discriminator.
                real = data.to(self.device)
                real_target = torch.full((b,), self.real_label, dtype=torch.float, device=self.device)
                
                real_output = self.bone.dis_run(real).view(-1)
                real_errD = self.criterion(real_output, real_target)
                real_errD.backward()
                real_D = real_output.mean().item()

                # Fake image on discriminator.
                z = torch.randn(b, 100, 1, 1, device=self.device)
                fake = self.bone.gen_run(z)
                fake_target = torch.full((b,), self.fake_label, dtype=torch.float, device=self.device)
                fake_output = self.bone.dis_run(fake.detach()).view(-1)
                fake_errD = self.criterion(fake_output, fake_target)
                fake_errD.backward()
                fake_D1 = fake_output.mean().item()

                errD = real_errD + fake_errD
                self.optimizer_dis.step()

                # Fake image on generator.
                self.freeze(self.bone.discriminator)
                g_fake_target = torch.full((b,), self.real_label, dtype=torch.float, device=self.device)
                fake_output = self.bone.dis_run(fake).view(-1)
                fake_errG = self.criterion(fake_output, g_fake_target)
                fake_errG.backward()
                fake_D2 = fake_output.mean().item()
                errG = fake_errG
                self.optimizer_gen.step()
                self.unfreeze(self.bone.discriminator)

                # Logging
                if (cur_iter % self.log_interval == 0):
                    tqdm.write("Iter {}".format(cur_iter))
                    tqdm.write("Real.D {:.4f} Fake.D.1 vs 2 {:.4f}:{:.4f} err.D {:.4f} err.G {:.4f}".format(real_D, fake_D1, fake_D2, errD.item(), errG.item()))

                # Save sample generated image
                if (cur_iter % self.save_interval == 0):
                    self.generate(cur_iter)
                    tqdm.write("Save to {}.pth".format(self.id))
                    torch.save(self.bone, "{}.pth".format(self.id))
                # Finish condition
                cur_iter += 1
                if (cur_iter >= self.iterations):
                    break
                pbar.update(1)
            if (cur_iter >= self.iterations):
                break

    def generate(self, iteration):
        """ Generating the sample images with fixed latent vector.
        """
        with torch.no_grad():
            fake_samples = self.bone.gen_run(self.sample_z)
            sample = vutils.make_grid(fake_samples.cpu().detach(), padding=2, normalize=True)
            sample = np.transpose(sample, (1, 2, 0))
            sample = sample.numpy()
            plt.imsave(os.path.join(self.dist_path, "{}.png".format(iteration)), sample)

def main(nz, ngf, ndf, nch, lr, iterations, batch_size, dataroot, device_id, run_id):
    device = torch.device(device_id)

    dataset = CelebA(dataroot)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)

    model = DCGAN(nz=nz, ngf=ngf, ndf=ndf, nch=nch)
    model = model.to(device)

    optimizer_gen = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_dis = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    # Never use SGD to optimize a GAN
    # optimizer_gen = torch.optim.SGD(model.generator.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    # optimizer_dis = torch.optim.SGD(model.discriminator.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)

    trainer = GANTrainer(run_id, model, dataloader, optimizer_gen, optimizer_dis, iterations=iterations, n_sample=64)
    trainer.train()

if __name__ == "__main__":
    RUN = 1
    
    nz = 100
    ngf = 64
    ndf = 64
    nch = 3
    lr = 1e-3
    iterations = 5000
    batch_size = 128
    # dataroot = os.environ["DATAROOT"]
    dataroot = "/home/huihui/Data"
    device_id = "cuda:0"
    main(nz, ngf, ndf, nch, lr, iterations, batch_size, dataroot, device_id, RUN)


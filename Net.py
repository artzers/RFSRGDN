import os, torch, tifffile
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as lrs

from scipy.ndimage.interpolation import zoom
import itertools
from torch.autograd import Variable
import torch.autograd as autograd
from models import AttentionUNet, DegradeNet, LowDiscriminator, HighDiscriminator

from Util import RestoreNetImg

logger = None
lowMean =0
lowStd = 0
highMean =0
highStd = 0
globalMax = 0

def get_style_loss(f1,f2):
    #same size
    _,c,d,h,w = f1.size()
    f1 = f1.view(c, d * h * w)
    gram1 = torch.mm(f1, f1.t())

    f2 = f2.view(c, d * h * w)
    gram2 = torch.mm(f2, f2.t())
    layer_style_loss = torch.mean((gram1 - gram2) ** 2)
    return layer_style_loss

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Trainer:
    def __init__(self,
                 data_loader,
                 test_loader,
                 scheduler=lrs.StepLR,
                 dev='cuda:0', devid=0):
        self.dataLoader = data_loader
        self.testLoader = test_loader
        self.dev = dev
        self.cudaid = devid

        # Loss function
        self.adversarial_loss = torch.nn.MSELoss()
        self.cycle_loss1 = torch.nn.SmoothL1Loss(reduction='mean')
        self.cycle_loss2 = torch.nn.SmoothL1Loss(reduction='mean')

        # Loss weights
        self.lambda_adv = 1
        self.lambda_cycle = 10
        self.lambda_gp = 10

        # Initialize generator and discriminator
        self.G_AB = AttentionUNet(in_channel = 1, num_class = 1)
        self.G_BA = DegradeNet()
        self.D_A = LowDiscriminator()
        self.D_B = HighDiscriminator()

        #self.G_AB.apply(weights_init_normal)
        self.G_BA.apply(weights_init_normal)
        self.D_A.apply(weights_init_normal)
        self.D_B.apply(weights_init_normal)
        # self.G_AB.load_state_dict(torch.load('./saved_models//G_AB_31500.pth'))
        # self.G_BA.load_state_dict(torch.load('./saved_models//G_BA_31500.pth'))
        # self.D_A.load_state_dict(torch.load('./saved_models//D_A_31500.pth'))
        # self.D_B.load_state_dict(torch.load('./saved_models//D_B_31500.pth'))

        self.G_AB.cuda(self.cudaid)
        self.G_BA.cuda(self.cudaid)
        self.D_A.cuda(self.cudaid)
        self.D_B.cuda(self.cudaid)

        # Optimizers
        lr = 0.00005
        self.optimizer_G = torch.optim.Adam([{'params': itertools.chain(self.G_AB.parameters(), \
                                                                        self.G_BA.parameters()), \
                                              'initial_lr': lr}], lr=lr)
        self.optimizer_D_A = torch.optim.RMSprop(params=[{'params':self.D_A.parameters(), \
                                                          'initial_lr':lr}], \
                                                   lr=lr)#RMSprop
        self.optimizer_D_B = torch.optim.RMSprop(params=[{'params': self.D_B.parameters(), \
                                                          'initial_lr': lr}], \
                                                 lr=lr)  # RMSprop

        self.scheduler_G = scheduler(self.optimizer_G, step_size=10000, gamma=0.9, last_epoch=-1)#36000
        self.scheduler_D_A = scheduler(self.optimizer_D_A, step_size=10000, gamma=0.9, last_epoch=-1)
        self.scheduler_D_B = scheduler(self.optimizer_D_B, step_size=10000, gamma=0.9, last_epoch=-1)

    def compute_gradient_penalty(self,D, real_samples, fake_samples, flag):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1, 1)))
        alpha = alpha.cuda(self.cudaid)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        validity = None
        if flag == True:
            validity,_,_,_ = D(interpolates)
        else:
            validity = D(interpolates)
        fake = Variable(torch.FloatTensor(np.ones(validity.shape)).cuda(self.cudaid), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=validity,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def Train(self, turn=2):
        self.shot = -1
        torch.set_grad_enabled(True)

        for t in range(turn):

            for kk, (lowImg, highImg) in enumerate(self.dataLoader):
                self.shot = self.shot + 1
                # torch.cuda.empty_cache()
                # self.scheduler.step()
                self.scheduler_G.step()
                self.scheduler_D_A.step()
                self.scheduler_D_B.step()
                lrImg = lowImg.cuda(self.cudaid)
                mrImg = highImg.cuda(self.cudaid)

                self.optimizer_D_A.zero_grad()
                self.optimizer_D_B.zero_grad()
                # Generate a batch of images
                fake_A = self.G_BA(mrImg).detach()
                fake_B = self.G_AB(lrImg).detach()
                # ----------
                # Domain A
                # ----------
                # Compute gradient penalty for improved wasserstein training
                gp_A = self.compute_gradient_penalty(
                    self.D_A, lrImg.data, fake_A.data, False)
                # Adversarial loss
                self.D_A_lrImg = torch.mean(self.D_A(lrImg))
                self.D_A_fakeA = torch.mean(self.D_A(fake_A))
                D_A_loss = -self.D_A_lrImg + self.D_A_fakeA + self.lambda_gp * gp_A
                # ----------
                # Domain B
                # ----------
                # Compute gradient penalty for improved wasserstein training
                gp_B = self.compute_gradient_penalty(self.D_B, mrImg.data, fake_B.data, True)
                # Adversarial loss
                self.D_B_mrImg,D_B_mrF1,D_B_mrF2,D_B_mrF3 = self.D_B(mrImg)
                self.D_B_mrImg = torch.mean(self.D_B_mrImg)
                self.D_B_fakeB,D_B_fakeF1,D_B_fakeF2,D_B_fakeF3 = self.D_B(fake_B)
                self.D_B_fakeB = torch.mean(self.D_B_fakeB)
                D_B_loss = -self.D_B_mrImg + self.D_B_fakeB + self.lambda_gp * gp_B
                #3D perceptual loss
                self.perceptLoss = ( 0.2 * get_style_loss(D_B_mrF1, D_B_fakeF1)
                + 0.3 * get_style_loss(D_B_mrF2, D_B_fakeF2)
                + 0.5 * get_style_loss(D_B_mrF3, D_B_fakeF3) )
                # Total loss
                D_loss = D_A_loss + D_B_loss
                final_D_loss = self.perceptLoss + D_loss

                final_D_loss.backward()
                torch.nn.utils.clip_grad_value_(self.D_A.parameters(), clip_value=1)
                torch.nn.utils.clip_grad_value_(self.D_B.parameters(),clip_value=1)
                torch.nn.utils.clip_grad_norm_(self.D_A.parameters(),max_norm=20, norm_type=2)
                torch.nn.utils.clip_grad_norm_(self.D_B.parameters(),max_norm=20, norm_type=2)
                self.optimizer_D_A.step()
                self.optimizer_D_B.step()

                if True:
                    # ------------------
                    #  Train Generators
                    # ------------------
                    self.optimizer_G.zero_grad()
                    # Translate images to opposite domain
                    fake_A = self.G_BA(mrImg)
                    fake_B = self.G_AB(lrImg)
                    # Reconstruct images
                    recov_A = self.G_BA(fake_B)
                    recov_B = self.G_AB(fake_A)
                    # Adversarial loss
                    G_adv = -torch.mean(self.D_A(fake_A)) - torch.mean(self.D_B(fake_B)[0])

                    lowCycle = 10 * self.cycle_loss1(
                        F.upsample(recov_A, scale_factor=(3, 3, 3)), F.upsample(lrImg, scale_factor=(3, 3, 3)))
                    highCycle = 10 * self.cycle_loss2(recov_B, mrImg)
                    # Total loss
                    G_loss = G_adv + lowCycle + highCycle

                    G_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.G_AB.parameters(), max_norm=20, norm_type=2)
                    torch.nn.utils.clip_grad_norm_(self.G_BA.parameters(), max_norm=20, norm_type=2)
                    self.optimizer_G.step()

                if self.shot % 20 == 0:

                    lr = self.scheduler_G.get_lr()[0]
                    lossVal = float(G_loss.cpu().data.numpy())
                    print("\r[Epoch %d] [Batch %d] [LR:%f] [D loss: %f] [G loss: %f, adv: %f,low cyc: %f, high cyc:%f, perp:%f]"
                          % (
                              t,
                              self.shot,
                              lr,
                              D_loss.item(),
                              G_loss.item(),
                              G_adv.item(),
                              lowCycle.item(),
                              highCycle.item(),
                              self.perceptLoss.item()
                          )
                          )

                    lrImgImg = np.max(lrImg.cpu().data.numpy()[0, 0, :, :, :], axis=0)
                    lrImgImg = RestoreNetImg(lrImgImg,0,1)
                    logger.img('lrXY', lrImgImg)
                    lrImgImg = np.max(lrImg.cpu().data.numpy()[0, 0, :, :, :], axis=1)
                    lrImgImg = RestoreNetImg(lrImgImg, 0, 1)
                    logger.img('lrXZ', lrImgImg)
                    recovImg = recov_B.cpu().data.numpy()[0, 0, :, :, :]
                    recovImg = RestoreNetImg(recovImg, 0, 1)
                    recovImg2XY = np.max(recovImg, axis=0)
                    recovImg2XZ = np.max(recovImg, axis=1)
                    logger.img('cycXY', recovImg2XY)
                    logger.img('cycXZ', recovImg2XZ)
                    reImg = fake_B.cpu().data.numpy()[0, 0, :, :, :]
                    reImg = RestoreNetImg(reImg, 0, 1)
                    reImg2XY = np.max(reImg, axis=0)
                    reImg2XZ = np.max(reImg, axis=1)
                    logger.img('srXY', reImg2XY)
                    logger.img('srXZ', reImg2XZ)
                    # interpolate
                    lrImg2 = lrImg.cpu().data.numpy()[0, 0, :, :, :]
                    zoom2 = RestoreNetImg(lrImg2, 0, 1)
                    zoom2 = np.minimum(zoom(zoom2, (3,3,3)), 255)

                    zoom2XY = np.max(zoom2, axis=0)
                    logger.img('z2XY', zoom2XY)
                    zoom2XZ = np.max(zoom2, axis=1)
                    logger.img('z2XZ', zoom2XZ)
                    highImgXY = np.max(highImg.data.numpy()[0, 0, :, :, :], axis=0)
                    highImgXY = RestoreNetImg(highImgXY, highMean, highStd)
                    logger.img('highXY', highImgXY)
                    highImgXZ = np.max(highImg.cpu().data.numpy()[0, 0, :, :, :], axis=1)
                    highImgXZ = RestoreNetImg(highImgXZ, highMean, highStd)
                    logger.img('highXZ', highImgXZ)
                    lossVal = float(G_loss.cpu().data.numpy())
                    if np.abs(lossVal) > 200:
                        print('G loss > 200')
                    else:
                        logger.plot('G_loss', lossVal)

                    lossVal = float(D_loss.cpu().data.numpy())
                    if np.abs(lossVal) > 200:
                        print('D loss > 200')
                    else:
                        logger.plot('D_loss', lossVal)
                if self.shot != 0 and self.shot % 500 == 0:
                    if not os.path.exists('saved_models/'):
                        os.mkdir('saved_models/')
                    torch.save(self.G_AB.state_dict(), "saved_models/G_AB_%d.pth" % ( self.shot))
                    torch.save(self.G_BA.state_dict(), "saved_models/G_BA_%d.pth" % ( self.shot))
                    torch.save(self.D_A.state_dict(), "saved_models/D_A_%d.pth" % ( self.shot))
                    torch.save(self.D_B.state_dict(), "saved_models/D_B_%d.pth" % ( self.shot))






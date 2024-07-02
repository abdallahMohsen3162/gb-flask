import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
import torch.nn.functional as F
import os
import gc
import time
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import VGG19_Weights
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg19
from torchvision.utils import save_image
import torch
import cv2
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from IPython.display import clear_output
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torchvision import transforms

torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf)
        self.rdb2 = ResidualDenseBlock(nf)
        self.rdb3 = ResidualDenseBlock(nf)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class Generator(nn.Module):
    def __init__(self, num_rrdb_blocks=23):
        super(Generator, self).__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.body = nn.Sequential(*[RRDB(64) for _ in range(num_rrdb_blocks)])
        self.conv_body = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.conv_up1 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.conv_up2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return feat

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output.view(output.size(0), -1)
      
      
class FeatureExtractor(nn.Module):
  def __init__(self):
    super(FeatureExtractor, self).__init__()
    vgg19_model = vgg19(pretrained=True)
    self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

  def forward(self, x):
    return self.feature_extractor(x)
  
  
  


class ESRGAN:
    def __init__(self, lr=1e-4, num_epochs=200, batch_size=16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.feature_extractor = FeatureExtractor().to(self.device)
        
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        
        self.content_criterion = nn.L1Loss()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def train(self, train_dataloader):
        for epoch in range(self.num_epochs):
            tqdm_dataloader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)
            for i, (lr_imgs, hr_imgs) in enumerate(tqdm_dataloader):
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                # Train Discriminator
                self.discriminator_optimizer.zero_grad()
                
                sr_imgs = self.generator(lr_imgs)
                real_preds = self.discriminator(hr_imgs)
                fake_preds = self.discriminator(sr_imgs.detach())
                
                d_loss_real = self.adversarial_criterion(real_preds - fake_preds.mean(), torch.ones_like(real_preds))
                d_loss_fake = self.adversarial_criterion(fake_preds - real_preds.mean(), torch.zeros_like(fake_preds))
                d_loss = (d_loss_real + d_loss_fake) / 2
                
                d_loss.backward()
                self.discriminator_optimizer.step()
                
                # Train Generator
                self.generator_optimizer.zero_grad()
                
                sr_imgs = self.generator(lr_imgs)
                fake_preds = self.discriminator(sr_imgs)
                real_preds = self.discriminator(hr_imgs)
                
                content_loss = self.content_criterion(sr_imgs, hr_imgs)
                adversarial_loss = self.adversarial_criterion(fake_preds - real_preds.mean(), torch.ones_like(fake_preds))
                perceptual_loss = self.content_criterion(self.feature_extractor(sr_imgs), self.feature_extractor(hr_imgs))
                
                g_loss = content_loss + 0.001 * adversarial_loss + 0.006 * perceptual_loss
                
                g_loss.backward()
                self.generator_optimizer.step()
                
                # Print or log detailed loss information for D and G
                if i % 100 == 0 or i==166:
                    d_loss_item = d_loss.item()
                    g_loss_item = g_loss.item()
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(train_dataloader)}], "
                          f"D_loss: {d_loss_item:.4f}, G_loss: {g_loss_item:.4f}, "
                          f"Content_loss: {content_loss.item():.4f}, "
                          f"Adversarial_loss: {adversarial_loss.item():.4f}, "
                          f"Perceptual_loss: {perceptual_loss.item():.4f}")
            
            tqdm_dataloader.close()
            

    def save_models(self, path):
        torch.save(self.generator.state_dict(), f"{path}/generator.pth")
        torch.save(self.discriminator.state_dict(), f"{path}/discriminator.pth")

    def load_models(self):
        generator_path = "generator.pth"
        discriminator_path = "discriminator.pth"

        # Determine the device to map the storage to
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.exists(generator_path):
            self.generator.load_state_dict(torch.load(generator_path, map_location=map_location))
            print(f"Generator model loaded successfully from {generator_path}.")
        else:
            print(f"Generator model file not found at {generator_path}. Skipping generator loading.")

        if os.path.exists(discriminator_path):
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=map_location))
            print(f"Discriminator model loaded successfully from {discriminator_path}.")
        else:
            print(f"Discriminator model file not found at {discriminator_path}. Skipping discriminator loading.")
              
              



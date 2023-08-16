import os
import copy
import argparse
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from utils import setup_logging, get_data, save_images, plot_images
from modules import EMA
from UNet_conditional import UNet_conditional

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self):
        """
        Linear Beta schedular is used in this version
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        """
        Forward Process
        
        Args:
            x (torch.float32): input images
            t (torch.int64): timestamp [batch_size,]

        Returns:
            noisy image: noisy image
            noise: The noise that is applied to each image [batch, 3, H, W]. This is the label to predict.
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.rand_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def sample_timesteps(self, n):
        """
        Generate timesteps

        Args:
            n (int): batch_size

        Returns:
            timestamp: [batch_size, ]
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n_image, labels, cfg_scale=3):
        """
        Reverse Process
        
        1. The model first sample a random noise from Gaussian Process. This can be considered as a extremely noisy image
        2. Then, the model predicts the noise based on the timestamp, and the generated noisey image above.
        3. Some predefined scaling parameters are retrieved by the timestamp.
        4. The noise is subtracted from the `extremely noisy image`. The result becomes the starting point for the next iteration. 
        5. Repeat. 
        
        Args:
            model (_type_): UNet
            n_image (int): the number of images that to generate

        Returns:
            x: the denoised images
        """
        logging.info(f"Sampling {n_image} new images ...")
        model.eval()
        with torch.no_grad():
            # import pdb;pdb.set_trace()
            x = torch.randn((n_image, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n_image) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.rand_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs"), args.run_name)
    l = len(dataloader)
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch: {epoch}")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MES", loss.item(), global_step=epoch * l + i)
            
        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n_image=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n_image=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Conditional"
    args.epochs = 1000
    # args.epochs = 1
    args.batch_size = 48
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = "/data/cifar10/cifar10-64/train"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)
    
def infer():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = "cuda"
    args.image_size = 64
    unet = UNet_conditional().to(args.device)
    diffusion = Diffusion(img_size=args.image_size, device=args.device)
    diffusion.sample(unet, 3)

if __name__ == "__main__":    
    launch()
    # infer()

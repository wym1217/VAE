import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import nn, optim
from vae import VAE
from model.model1 import Encoder1,Decoder1
from log import loss_BCE, loss_KLD, latent_size, hidden_size, input_size, output_size, epochs, learning_rate, device
from tqdm import tqdm   
import numpy as np
import matplotlib.pyplot as plt  
dim = 2

if dim == 1:
    model_path = 'dim1.pth'
    model = VAE(input_size,output_size,latent_size = 1,hidden_size = 128).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    n = 20

    z_values = torch.linspace(-5, 5, steps=100).unsqueeze(1)  # shape: [10, 1]
    with torch.no_grad():
        generated_imgs = model.decoder(z_values.to(device)).cpu()  # shape: [10, 1, 28, 28]

    # 可视化
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_imgs[i].view(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f'z={z_values[i].item():.1f}')
    plt.tight_layout()
    plt.show()

elif dim == 2:
    model_path = 'dim2.pth'
    model = VAE(input_size,output_size,latent_size = 2,hidden_size = 128).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 创建 2D 网格：从 -3 到 3，10×10 共 100 个点
    grid_x = torch.linspace(-5, 5, steps=10)
    grid_y = torch.linspace(-5, 5, steps=10)
    z_list = []

    for y in grid_y:
        for x in grid_x:
            z_list.append(torch.tensor([x, y]))

    z_tensor = torch.stack(z_list).to(torch.float32)  # shape: [100, 2]

    # 使用已训练好的 decoder
    
    with torch.no_grad():
        generated_imgs = model.decoder(z_tensor.to(device)).cpu()  # [100, 1, 28, 28]

    # 绘制图像
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img = generated_imgs[i].view(28, 28) # shape: [1, 28, 28] -> [28, 28]
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
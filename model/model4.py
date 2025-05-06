import torch
from torch import nn
class ConvEncoder(nn.Module):
    def __init__(self, latent_size=16):
        super(ConvEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_size)
        self.fc_sigma = nn.Linear(64 * 7 * 7, latent_size)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # Ensure correct shape
        x = self.conv(x)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma

class ConvDecoder(nn.Module):
    def __init__(self, latent_size=16):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(latent_size, 64*7*7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 7, 7)
        x = self.deconv(x)
        return x
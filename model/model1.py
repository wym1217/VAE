import torch
from torch import nn
class Encoder1(nn.Module):
    def __init__(self, input_size = 28*28, hidden_size = 128, latent_size = 16):
        super(Encoder1, self).__init__()
        # Encoder里，我尝试了3层线性层、ReLU激活函数.
        self.linear = nn.Sequential(   
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_size),
            nn.ReLU()
        ) # 使用nn.Sequential将多个层组合在一起
        self.mu = nn.Linear(hidden_size, latent_size)
        self.sigma = nn.Linear(hidden_size, latent_size)
    def forward(self, x):
        y = self.linear(x)
        mu = self.mu(y) 
        sigma = self.sigma(y)
        return mu,sigma

class Decoder1(nn.Module):
    def __init__(self, latent_size = 16, hidden_size = 128, output_size = 28*28):
        super(Decoder1, self).__init__()
        # Decoder也是同样对称的设置.
        self.linear1 = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_size),
            nn.ReLU(),
        )
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, x): 
        y = self.linear1(x)
        z = torch.sigmoid(self.linear2(y)) # 输出限制在0-1之间，适合于图像数据
        return z
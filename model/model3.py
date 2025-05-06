import torch
from torch import nn
class Encoder3(nn.Module):
    def __init__(self, input_size = 28*28, hidden_size = 128, latent_size = 16):
        super(Encoder3, self).__init__()
        # 仅用一层线性层，最原始的设计。
        self.linear = nn.Linear(input_size, hidden_size)
        self.mu = nn.Linear(hidden_size, latent_size)
        self.sigma = nn.Linear(hidden_size, latent_size)
    def forward(self, x):
        y = self.linear(x)
        z = torch.relu(y) # 激活函数
        mu = self.mu(z) 
        sigma = self.sigma(z)
        return mu,sigma

class Decoder3(nn.Module):
    def __init__(self, latent_size = 16, hidden_size = 128, output_size = 28*28):
        super(Decoder3, self).__init__()
        # decoder同理。
        self.linear1 = nn.Linear(latent_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, x): 
        y = self.linear1(x)
        z = torch.sigmoid(self.linear2(y)) 
        return z
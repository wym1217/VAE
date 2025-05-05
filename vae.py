import torch
from torch import nn
from model.model1 import Encoder1,Decoder1
from model.model2 import Encoder2,Decoder2
from model.model3 import Encoder3,Decoder3
from model.model4 import ConvEncoder,ConvDecoder

class VAE(torch.nn.Module):
    #将编码器解码器组合
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(VAE, self).__init__()
        # self.encoder = ConvEncoder(input_size, hidden_size, latent_size)
        # self.decoder = ConvDecoder(latent_size, hidden_size, output_size)
        self.encoder = ConvEncoder(latent_size)
        self.decoder = ConvDecoder(latent_size)
        # self.encoder = Encoder1(input_size, hidden_size, latent_size)
        # self.decoder = Decoder1(latent_size, hidden_size, output_size)

    def encode(self, x):
        mu,sigma = self.encoder(x)
        return mu,sigma
    
    def reparameterize(self, mu, sigma): # 采样
        eps = torch.randn_like(sigma)  
        z = mu + eps*sigma  # 利用正太分布采样
        return z
    
    def decode(self, z):
        re_x = self.decoder(z) 
        return re_x
    
    def forward(self, x):
        mu,sigma = self.encoder(x) 
        z = self.reparameterize(mu, sigma)  
        re_x = self.decoder(z) 
        return re_x,mu,sigma
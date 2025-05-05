import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import nn, optim
from vae import VAE
from log import loss_BCE, loss_KLD, latent_size, hidden_size, input_size, output_size, epochs, learning_rate, device
from tqdm import tqdm                                                                                                                               

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

batch_size=256
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

modelname = 'best.pth'
model = VAE(input_size,output_size,latent_size,hidden_size).to(device)
use_model = True
if use_model:
    model.load_state_dict(torch.load(modelname))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
min_train_loss = 1e10
min_test_loss = 1e10
#train
beta = 2
loss_history = {'train':[],'eval':[]}
for epoch in range(epochs):   
    #训练
    model.train()
    #每个epoch重置损失，设置进度条
    train_loss = 0
    train_nsample = 0
    t = tqdm(train_loader,desc = f'[train]epoch:{epoch}')
    for imgs, lbls in t: 
        bs = imgs.shape[0]
        imgs = imgs.to(device).view(bs,input_size)   
        re_imgs, mu, sigma = model(imgs)
        # loss_re = loss_BCE(re_imgs, imgs)
        loss_re = loss_BCE(re_imgs.view(-1, 784), imgs.view(-1, 784))
        loss_norm = loss_KLD(mu, sigma) # 正态分布(mu,sigma)与正态分布(0,1)的差距
        # loss = loss_re + beta * loss_norm
        loss = loss_re + loss_norm

        #反向传播、参数优化，重置
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #计算平均损失，设置进度条
        train_loss += loss.item()
        train_nsample += bs
        t.set_postfix({'loss':train_loss/train_nsample})
    #每个epoch记录总损失
    loss_history['train'].append(train_loss/train_nsample)
    if train_loss/train_nsample < min_train_loss:
        min_train_loss = train_loss/train_nsample

    #测试
    model.eval()
    #每个epoch重置损失，设置进度条
    test_loss = 0
    test_nsample = 0
    e = tqdm(test_loader,desc = f'[eval]epoch:{epoch}')
    for imgs, label in e:
        bs = imgs.shape[0]
        imgs = imgs.to(device).view(bs,input_size)  
        re_imgs, mu, sigma = model(imgs)
        # loss_re = loss_BCE(re_imgs, imgs) 
        loss_re = loss_BCE(re_imgs.view(-1, 784), imgs.view(-1, 784))
        loss_norm = loss_KLD(mu, sigma) 
        # loss = loss_re + beta * loss_norm
        loss = loss_re + loss_norm
        #计算平均损失，设置进度条
        test_loss += loss.item()
        test_nsample += bs
        e.set_postfix({'loss':test_loss/test_nsample})
    #每个epoch记录总损失    
    loss_history['eval'].append(test_loss/test_nsample)
    if test_loss/test_nsample < min_test_loss:
        min_test_loss = test_loss/test_nsample
new_model = '1.pth'
torch.save(model.state_dict(),new_model)

#plot loss
import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.arange(epochs),loss_history['train'],label = 'train')
plt.plot(np.arange(epochs),loss_history['eval'],label = 'eval')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss history')
plt.savefig('loss.png')
plt.show()



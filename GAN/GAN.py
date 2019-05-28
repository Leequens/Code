import argparse
import os
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch

parser= argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--img_size', type=int, default=28)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--samples_save_freq', type=int, default=500)
opt = parser.parse_args()
print(opt)
class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(opt.dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.model(x)
        img= x.view(x.size(0), opt.channels, opt.img_size, opt.img_size)
        return img

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x=x.view(x.size(0),-1)
        out=self.model(x)
        return out

Gen=G()
Dis=D()
loss=torch.nn.BCELoss()
Gen.cuda()
Dis.cuda()
loss.cuda()
optimizer_G=torch.optim.Adam(Gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D=torch.optim.Adam(Dis.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
data = datasets.MNIST(
    "../../data/mnist",
    download=False,
    train=True,
    transform=transforms.Compose(
        [transforms.Resize(opt.img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])]
    )
)
dataloder = DataLoader(data, batch_size=opt.batch_size, shuffle=True)
Tensor = torch.cuda.FloatTensor
for epoch in range(opt.epochs):
    for i, (imgs, _) in enumerate(dataloder):
        true = Variable(Tensor(imgs.size(0), 1).fill_(1.0))
        false =  Variable(Tensor(imgs.size(0), 1).fill_(0.0))
        real_img = Variable(imgs.type(Tensor))
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.dim))))
        optimizer_G.zero_grad()
        fake_imgs = Gen(z)
        g_loss = loss(Dis(fake_imgs), true)
        g_loss.backward()
        optimizer_G.step()
        optimizer_D.zero_grad()
        real_loss = loss(Dis(real_img), true)
        fake_loss = loss(Dis(fake_imgs.detach()), false) # .detach()用于截断梯度，因为fake_imgs是从G的forward产生的，这里只想更新loss_D而不影响G
        d_loss = (real_loss+fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        print('Epoch[%d/%d], BATCH[%d/%d], D_loss: %f, G_loss : %f'
              %( epoch, opt.epochs, i, len(dataloder), d_loss.item(), g_loss.item()))
        batch=epoch*len(dataloder)+i
        if batch % opt.samples_save_freq==0:
            save_image(fake_imgs.data[:25], 'GAN/%d.png' % batch, nrow=5, normalize=True)
    torch.save(Gen.state_dict(), 'GAN/Gen_epoch_%d.pkl' %epoch)

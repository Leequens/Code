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
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--img_size', type=int, default=28)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--samples_save_freq', type=int, default=500)
parser.add_argument('--model_dir', type=str, default='GAN1/Gen_epoch_199.pkl')

opt = parser.parse_args()
print(opt)
if os.path.exists('GAN/test/real') is False:
    os.makedirs('GAN/test/real')
if os.path.exists('GAN/test/fake') is False:
    os.makedirs('GAN/test/fake')
class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(opt.dim, 128),
           # nn.BatchNorm1d(128),
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

data=datasets.MNIST(
    '../../data/mnist',
    train=False,
    transform=transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.50])
    ]),
    download=False
)
dataloader=DataLoader(data,
                     batch_size=opt.batch_size,
                     shuffle=False)
Tensor = torch.cuda.FloatTensor

Gen = G()
Gen.cuda()
Gen.load_state_dict(torch.load(opt.model_dir))
Gen.eval()  #将神经网络固定。指定为测试状态，训练状态用model.train()来指定。

for i, (imgs, _) in enumerate(dataloader):
    z=Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.dim))))
    real_img = Variable(imgs.type(Tensor))
    fake_img = Gen(z)
    print("[Batch %d/%d]" % (i, len(dataloader)))
    real_img = real_img.view(real_img.shape[0], 1, 28, 28)
    fake_img = fake_img.view(fake_img.shape[0], 1, 28, 28)

    save_image(real_img.data, 'GAN1/test/real/real_img_%d.png' % i, nrow=1, normalize=True)
    save_image(fake_img.data, 'GAN1/test/fake/fake_img_%d.png' % i, nrow=1, normalize=True)

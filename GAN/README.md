##pytorch-GAN
- Generator: 100→128→BachNormal→256→BachNormal→512→BachNormal→1024→786
- Discriminator: 786→512→256→1
- epoch:200
- 评价指标: Pytorch-FID
- 测试时fake imgaes和real images均为10000张 。
- 测得FID为：38.4

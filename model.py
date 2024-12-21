import matplotlib.pyplot as plt
import torch
import torchvision
from sympy.core.rules import Transform
from torch import nn
from torchvision import transforms
from setting import datasetup, engine
from setting.helper_functions import set_seeds, plot_loss_curve

# 训练参数设置
BATCH_SIZE = 32
IMG_SIZE = 224
PATCH_SIZE = 16
# 设置训练设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 以烟草作为数据集,并且录入dataloader
train_dir = "C://Users//Administrator//Desktop//TOBACCO20241211//train"
val_dir = "C://Users//Administrator//Desktop//TOBACCO20241211//val"
Tabacco_Transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
train_loader, val_loader, class_names = datasetup.create_dataloaders(train_dir=train_dir, val_dir=val_dir,
                                                                     transform=Tabacco_Transform, batch_size=BATCH_SIZE)

# 将图像转化为vit需要的patch
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = PATCH_SIZE, embed_dim: int = 768):
        super(PatchEmbedding, self).__init__()
        # 将数据嵌入到patch中，最好的patch size在论文中给出为16*16
        # 通过kernel size 为16的卷积来实现对图片的patch序列化处理  设置 in_channels=3 作为图像中颜色通道的数量，并设置 out_channels=768 ，与 表 1 中 ViT-Base 的值（这是嵌入维度，每个图像将被嵌入到大小为 768 的可学习向量中）
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size,
                                 stride=patch_size, padding=0)
        # 将卷积得到的2d格式转化为嵌入层输出需要的矩阵模式[1,768,14,14]->[1,768,196]
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)  # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]

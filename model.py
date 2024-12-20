import matplotlib.pyplot as plt
import torch
import torchvision
from sympy.core.rules import Transform
from torch import nn
from torchvision import transforms
from setting import datasetup, engine
from setting.helper_functions import set_seeds, plot_loss_curve

# 训练参数设置
BATCH_SIZE=32
IMG_SIZE=224

#设置训练设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 以烟草作为数据集,并且录入dataloader
train_dir="C://Users//Administrator//Desktop//TOBACCO20241211//train"
val_dir="C://Users//Administrator//Desktop//TOBACCO20241211//val"
Tabacco_Transform=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
])
train_loader,val_loader,class_names = datasetup.create_dataloaders(train_dir=train_dir,val_dir=val_dir,transform=Tabacco_Transform,batch_size=BATCH_SIZE)

# 将数据嵌入到patch中，最好的patch size在论文中给出为16*16
# 通过kernel size 为16的卷积来实现对图片的patch序列化处理


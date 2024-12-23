import matplotlib.pyplot as plt
import torch
import torchvision
from mpl_toolkits.mplot3d.proj3d import transform
from oauthlib.uri_validate import query
from sympy.core.rules import Transform
from torchinfo import summary
from torch import nn
from torch.nn.functional import embedding
from torchvision import transforms
from setting import datasetup, engine
from setting.helper_functions import set_seeds, plot_loss_curve

# 训练参数设置
BATCH_SIZE = 64
IMG_SIZE = 224
PATCH_SIZE = 16
IN_CHANNELS = 3
NUM_TRANSFOMER_LAYER = 12
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
                                                                     transform=Tabacco_Transform, batch_size=BATCH_SIZE,num_workers=4)

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

# # 设置class token
# # torch.Size([32, 1, 768]) -> [batch_size, number_of_tokens, embedding_dimension]")
# class_token = nn.Parameter(torch.ones(BATCH_SIZE,1,768),requires_grad=True)
#
# # 创建位置嵌入保留位置信息
# # torch.Size([32, 197, 768]) -> [batch_size, number_of_patches+class_token, embedding_dimension]")
# number_of_patches = 196
# position_embedding = nn.Parameter(torch.ones(BATCH_SIZE,number_of_patches+1,768),requires_grad=True)
# # 直接和序列相加即可

# 创建多头自注意力模型
class MultiheadSelfAttention(nn.Module):
    def __init__(self,  embed_dim: int=768, num_heads: int=12,attn_dropout: float = 0):
        super(MultiheadSelfAttention, self).__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,dropout=attn_dropout,batch_first=False)

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        return attn_output

# MLP 包括layer norm->linear->nonlinear->dropout->linear->dropout,维度先扩展再压缩，整体与注意力层保持不变
class MLPBlock(nn.Module):
    def __init__(self, embed_dim: int=768, mlp_size: int=3072,dropout: float = 0.1):
        super(MLPBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=mlp_size),
                                 nn.GELU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(in_features=mlp_size, out_features=embed_dim),
                                 nn.Dropout(p=dropout))
    def forward(self, x):
        x=self.layer_norm(x)
        x = self.mlp(x)
        return x

# 创建encoder块
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0):
        super(TransformerEncoderBlock,self).__init__()
        self.msa_block=MultiheadSelfAttention(embed_dim=embedding_dim,num_heads=num_heads,attn_dropout=attn_dropout)
        self.mlp_block=MLPBlock(embed_dim=embedding_dim,mlp_size=mlp_size,dropout=mlp_dropout)

    def forward(self,x):
        # 记得进行残差处理
        x=self.msa_block(x)+x
        x=self.mlp_block(x)+x
        return x

# transformer_encoder_blocks=TransformerEncoderBlock()
# summary(model=transformer_encoder_blocks,
#         input_size=(1,197,768),
#         col_names=["input_size","output_size","num_params","trainable"],
#         col_width=20,
#         row_settings=["var_names"])
#
class Vit(nn.Module):
    def __init__(self,
                 image_size:int=IMG_SIZE,
                 in_channels:int=IN_CHANNELS,
                 patch_size:int=PATCH_SIZE,
                 num_transformer_layers:int=NUM_TRANSFOMER_LAYER,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=12,
                 attn_dropout:float=0,
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 num_classes:int=len(class_names)):
        super(Vit,self).__init__()
        self.num_patches = (image_size * image_size) // patch_size**2
        # 分类嵌入
        self.class_embedding = nn.Parameter(data=torch.randn(1,1,embedding_dim),requires_grad=True)
        # 位置嵌入
        self.position_embedding = nn.Parameter(data=torch.randn(1,self.num_patches+1,embedding_dim),requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels,patch_size=patch_size,embed_dim=embedding_dim)

        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           attn_dropout=attn_dropout,
                                                                           mlp_dropout=mlp_dropout
                                                                           ) for _ in range(num_transformer_layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim),
                                        nn.Linear(in_features=embedding_dim,
                                                  out_features=num_classes))

    def forward(self,x):
        batch_size = x.size(0)
        class_token = self.class_embedding.expand(batch_size,-1,-1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token,x),dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:,0])
        return x

vit = Vit(num_classes=len(class_names))
# summary(model=vit,
#         input_size=(32, 3, 224, 224),  # (batch_size, color_channels, height, width)
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
#         )

"""
============================================================================================================================================
Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
============================================================================================================================================
Vit (Vit)                                                    [32, 3, 224, 224]    --                   152,064              True
├─PatchEmbedding (patch_embedding)                           [32, 3, 224, 224]    [32, 196, 768]       --                   True
│    └─Conv2d (patcher)                                      [32, 3, 224, 224]    [32, 768, 14, 14]    590,592              True
│    └─Flatten (flatten)                                     [32, 768, 14, 14]    [32, 768, 196]       --                   --
├─Dropout (embedding_dropout)                                [32, 197, 768]       [32, 197, 768]       --                   --
├─Sequential (transformer_encoder)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    └─TransformerEncoderBlock (0)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (1)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (2)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (3)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (4)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (5)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (6)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (7)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (8)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (9)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (10)                          [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (11)                          [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiheadSelfAttention (msa_block)               [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
├─Sequential (classifier)                                    [32, 768]            [32, 4]              --                   True
│    └─LayerNorm (0)                                         [32, 768]            [32, 768]            1,536                True
│    └─Linear (1)                                            [32, 768]            [32, 4]              3,076                True
============================================================================================================================================
Total params: 85,801,732
Trainable params: 85,801,732
Non-trainable params: 0
Total mult-adds (G): 5.52
============================================================================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 3292.20
Params size (MB): 229.21
Estimated Total Size (MB): 3540.67
============================================================================================================================================

"""
from setting import engine

# Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper
optimizer = torch.optim.Adam(params=vit.parameters(),
                             lr=1e-3,  # Base LR from Table 3 for ViT-* ImageNet-1k
                             betas=(0.9, 0.999),
                             # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                             weight_decay=0.01)  # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

# Setup the loss function for multi-class classification
loss_fn = torch.nn.CrossEntropyLoss()

# Set the seeds
set_seeds()
# Train the model and save the training results to a dictionary
if __name__ == '__main__':
    # 你的 multiprocessing 代码或者任何会创建新进程的代码
    results = engine.train(model=vit,
                           train_loader=train_loader,
                           val_loader=val_loader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=10,
                           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    plot_loss_curve(results)


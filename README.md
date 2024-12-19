# vision-transformer
复现vit

vision transformer旨在原始的transformer架构适应视觉问题


进行设置：引入基础库，以及引入前面章节可以重复使用的函数。

获取数据：和前面博客保持一致，使用的披萨、牛排和寿司图像分类数据集，并构建一个 Vision Transformer 来尝试改进 FoodVision Mini 模型的结果。
创建Dataset和DataLoader：重复使用data_setup.py 脚本来设置我们的 DataLoaders。

复现ViT论文

Equation 1: The Patch Embedding：ViT 架构由四个主要公式组成，第一个是 patch 和位置嵌入。或者将图像转换为一系列可学习的 patch 。

Equation 2: Multi-Head Attention (MSA)【多头注意力】：自注意力/多头自注意力（MSA）机制是每个 Transformer 架构（包括 ViT 架构）的核心，使用 PyTorch 的内置层创建一个 MSA 块。

Equation 3: Multilayer Perceptron (MLP)【多层感知机】：ViT 架构使用多层感知器作为其 Transformer Encoder 的一部分及其输出层。首先为 Transformer Encoder 创建 MLP。

创建 Transformer 编码器（encode）：Transformer 编码器通常由通过残差连接连接在一起的 MSA（公式 2）和 MLP（公式 3）的交替层组成。
将它们放在一起创建 ViT

为 ViT 模型设置训练代码：可以重复使用前面博客的engine.py 中的 train() 函数

使用来自 torchvision.models 的预训练 ViT ：训练像 ViT 这样的大型模型通常需要大量数据。

对自定义图像进行预测

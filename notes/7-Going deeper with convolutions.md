# 7-Going deeper with convolutions

## 1. 要解决的问题

传统的提高神经网络性能的方法：

- 增加深度（`网络层次数量`）
- 增加宽度（`每一层的神经元数量`）

该方案简单、容易但有两个主要的缺点：

- 容易过拟合

尺寸增大则参数数量增加，使网络更容易过拟合，尤其是样本不足的情况下。

- 需要更强的计算能力

均匀增加网络尺寸会使得计算资源使用的显著增加。例如，在一个深度视觉网络中，如果两个卷积层相连，它们的滤波器数目的任何均匀增加都会引起计算量平方式的增加。如果增加的能力使用时效率低下（例如，如果大多数权重最后接近于0），那么会浪费大量的计算能力。

## 2. 过去的解决方法及问题

### 2.1 稀疏连接

解决以上两个问题的一个基本的方式就是引入稀疏性，将全连接层替换为稀疏的全连接层，甚至是卷积层。（理论依据来源 Provablebounds for learning some deep representations：https://arxiv.org/abs/1310.6343）

稀疏连接有两种方法：

1. 空间（`spatial`）上的稀疏连接，也就是 `CNN` 。其只对输入图像的局部进行卷积，而不是对整个图像进行卷积，同时参数共享降低了总参数的数目并减少了计算量
2. 在特征（`feature`）维度上的稀疏连接进行处理，也就是在通道的维度上进行处理。

**问题：稀疏层计算能力浪费**

非均匀的稀疏模型要求更多的复杂工程和计算基础结构,当碰到在非均匀的稀疏数据结构上进行数值计算时，现在的计算架构效率非常低下。

## 3. 提出的新知识

针对以上情况，我们希望找到一种方法**既能保持网络结构的稀疏性，又能利用密集矩阵的高计算性能**。大量的文献表明可以将稀疏矩阵聚类为较为密集的子矩阵来提高计算性能，据此论文提出了名为`Inception `的结构来实现此目的。

### 3.1 知识补充

#### 3.1.1 稀疏矩阵的分解

如下图所示，为稀疏矩阵分解为密集矩阵运算的例子，可以发现，运算量确实降低不少。

![img](https://pic3.zhimg.com/v2-cee9d33c6f93101609ab7f392ca5f016_r.jpg)

#### 3.1.2 filter bank 维度处理

上面系数矩阵对应到 Inception 中，就是在通道维度上，将稀疏连接转换为密集连接。

比方说 3×3 的卷积核，提取 256 个特征，其总特征可能会均匀分散于每个 feature map上，可以理解为一个稀疏连接的特征集。可以极端假设，64 个 filter 提取的特征的密度，显然比 256个 filter 提取的特征的密度要高。

因此，通过使用$1×1、3×3、5×5 $等，卷积核数分别为 96, 96, 64 个，分别提取不同尺度的特征，并保持总的 filter bank 尺寸不变。这样，同一尺度下的特征之间的相关性更强，密度更大，而不同尺度的特征之间的相关性被弱化。

综上所述，可以理解为将一个包含 256 个均匀分布的特征，分解为了几组强相关性的特征组。同样是 256 个特征，但是其输出特征的冗余信息更少。

> 参考博客： https://zhuanlan.zhihu.com/p/69345065 Inception-v1 论文详解

## 4&5. 重要的相关工作（关键方法）

### 4.1 Inception module

**Inception** 结构的主要思路是怎样用密集成分来近似最优的局部稀疏结构。

作者首先提出下图这样的基本结构：

<img src="https://upload-images.jianshu.io/upload_images/24408091-23881deb5e2aa43b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" style="zoom:80%;" />

对上图做以下说明：

1. 采用不同大小的卷积核意味着不同大小的感受野，最后拼接意味着不同尺度特征的融合；

2. 之所以卷积核大小采用1、3和5，主要是为了方便对齐。设定卷积步长stride=1之后，只要分别设定pad=0、1、2，那么卷积之后便可以得到相同维度的特征，然后这些特征就可以直接拼接在一起了；

3. 文章说很多地方都表明pooling挺有效，所以Inception里面也嵌入了。

4. 网络越到后面，特征越抽象，而且每个特征所涉及的感受野也更大了，因此随着层数的增加，3x3和5x5卷积的比例也要增加。

**但是，使用5x5的卷积核仍然会带来巨大的计算量。** 为此，文章采用1x1卷积核来进行**降维**。
例如：上一层的输出为100x100x128，经过具有256个输出的5x5卷积层之后(stride=1，pad=2)，输出数据为100x100x256。其中，卷积层的参数为128x5x5x256。假如上一层输出先经过具有32个输出的1x1卷积层，再经过具有256个输出的5x5卷积层，那么最终的输出数据仍为为100x100x256，但卷积参数量已经减少为128x1x1x32 + 32x5x5x256，大约减少了4倍。

具体改进后的Inception Module如下图：

<img src="https://upload-images.jianshu.io/upload_images/24408091-6461091ef01a9315.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" style="zoom:80%;" />

> 参考博客：https://blog.csdn.net/shuzfan/article/details/50738394 GoogLeNet系列解读
>

### 4.2 GoogLeNet

#### 4.2.1 主网络配置

![](https://upload-images.jianshu.io/upload_images/24408091-2a7a9e19d06fcdf9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在 `GoogLeNet` 中：

1. 输入尺寸为$224×224$ ，使用 `RGB` 格式的图像，并分别对每个通道减去各通道的均值。
2. 所有的激活函数使用 `ReLU`
3. 上表中，其中，$\#3×3$和$\#5×5$分别表示在对应卷积核之前使用的$1×1$的卷积和通道数。
4. 网络设计始终考虑在移动设备上的实用性，最终网络深度为 `22` 层 (不包含池化层)。总的网络层数大约为 `100` 层。
5. 当将最后的全连接层，替换为均值池化层，将会使得 `top-1` 精度改善 `0.6%`。因为全连接层占据大部分计算量，且容易过拟合。（实际在最后还是加了一个全连接层，主要是为了方便以后大家finetune）
6. 虽然移除了全连接，但是网络中依然使用了`Dropout`

#### 4.2.2 辅助分类器 auxiliary classifiers

为解决梯度消失问题、归一化网络，额外增加了2个辅助分类器用于向前传导梯度。（实际上作用不大）

辅助分类器由小的 `CNNs` 组成，放置在Inception (4a)和Inception (4b)模块的输出之上。在训练期间，它们的损失以折扣权重（辅助分类器损失的权重是0.3）加到网络的整个损失上。实际测试的时候，这两个额外分支会被去掉。

#### 4.2.3 最终模型

最终的模型结构如下图所示：

![](https://upload-images.jianshu.io/upload_images/24408091-3b51ff29b3f1a460.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 6&7&8. 实验的设计

### 6.1 ILSVRC 2014分类挑战赛设置和结果

ILSVRC 2014分类挑战赛包括将图像分类到ImageNet层级中1000个叶子结点类别的任务。训练图像大约有120万张，验证图像有5万张，测试图像有10万张。每一张图像与一个实际类别相关联，性能度量基于分类器预测的最高分。

#### 6.1.1 评价指标

**top-1accuracy rate**: 比较实际类别和第一个预测类别

**top-5 error rate**: 比较实际类别与前5个预测类别：如果图像实际类别在top-5中，则认为图像分类正确，不管它在top-5中的排名。

挑战赛使用`top-5 error rate`来进行排名。

#### 6.1.2 训练 

- 没有使用外部数据来训练。
- GoogLeNet网络使用DistBelief[4]分布式机器学习系统进行训练，该系统使用适量的模型和数据并行。
- 使用异步随机梯度下降，动量参数为0.9[17]，固定的学习率计划（每8次遍历下降学习率4%）。
- 一些模型主要是在相对较小的裁剪图像进行训练，其它模型主要是在相对较大的裁剪图像上进行训练。
- 各种尺寸的图像块的采样，它的尺寸均匀分布在图像区域的8%——100%之间，方向角限制为$\left[\frac{3}{4}, \frac{4}{3}\right]$之间。
- Andrew Howard[8]的光度扭曲克服训练数据成像条件的过拟合。

#### 6.1.3 测试

为获得更好的测试效果，采用了一系列技巧（emsemble&crop）：

1. 独立训练了7个版本的相同的GoogLeNet模型，并用它们进行了整体预测。这些模型的训练具有相同的初始化和学习率策略，仅在采样方法和随机输入图像顺序方面不同。

2. 在测试中，采用更激进的裁剪方法。

   具体来说，我们将图像归一化为四个尺度，其中较短维度（高度或宽度）分别为256，288，320和352，取这些归一化的图像的左，中，右方块（在肖像图片中，我们采用顶部，中心和底部方块）。对于每个方块，我们将采用4个角以及中心224×224裁剪图像以及方块尺寸归一化为224×224，以及它们的镜像版本。这导致每张图像会得到4×3×6×2 = 144的裁剪图像。

   在实际应用中，这种激进的裁剪可能是不必要的，因为存在合理数量的裁剪图像后，更多裁剪图像的好处会变得很微小。

3. softmax概率在多个裁剪图像上和所有单个分类器上进行平均，然后获得最终预测。

   分析了验证数据的替代方法，例如裁剪图像上的最大池化和分类器的平均，但是它们比简单平均的性能略逊。

#### 6.1.4 结果

![](https://upload-images.jianshu.io/upload_images/24408091-dd903cb9437a65ab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最终在验证集和测试集上得到了`top-5 6.67%`的错误率，在其它的参与者中排名第一。表2显示了过去三年中一些表现最好的方法的统计。

![image-20201121154717904](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20201121154717904.png)

表3中分析报告了多种测试选择的性能，当预测图像时通过改变使用的模型数目和裁剪图像数目。

### 6.2 ILSVRC 2014检测挑战赛设置和结果

ILSVRC检测任务是为了在200个可能的类别中生成图像中目标的边界框。如果检测到的对象匹配的它们实际类别并且它们的边界框重叠至少50%（使用Jaccard索引），则将检测到的对象记为正确。无关的检测记为假阳性且被惩罚。

#### 6.2.1 评价指标

mAP（mean average precision)平均精度均值

#### 6.2.2 使用的技巧

GoogLeNet检测采用的方法类似于R-CNN[6]，但用Inception模块作为区域分类器进行了增强。

此外，为了更高的目标边界框召回率，通过选择搜索[20]方法和多箱[5]预测相结合改进了区域生成步骤。

为了减少假阳性的数量，超分辨率的尺寸增加了2倍。这将选择搜索算法的区域生成减少了一半。

我们总共补充了200个来自多盒结果的区域生成，大约60%的区域生成用于[6]，同时将覆盖率从92%提高到93%。减少区域生成的数量，增加覆盖率的整体影响是对于单个模型的情况平均精度均值增加了1%。

最后，等分类单个区域时，我们使用了6个GoogLeNets的组合。这导致准确率从40%提高到43.9%。注意，与R-CNN相反，由于缺少时间我们没有使用边界框回归。

#### 6.2.3 结果

![](https://upload-images.jianshu.io/upload_images/24408091-3ffe7a4ad39fa0f9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在表4中报告了官方的分数和每个队伍的常见策略：使用外部数据、集成模型或上下文模型。

外部数据通常是ILSVRC12的分类数据，用来预训练模型，后面在检测数据集上进行改善。一些团队也提到使用定位数据。由于定位任务的边界框很大一部分不在检测数据集中，所以可以用该数据预训练一般的边界框回归器，这与分类预训练的方式相同。GoogLeNet输入没有使用定位数据进行预训练。

![](https://upload-images.jianshu.io/upload_images/24408091-9fd491a3b3bc856a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在表5中，我们仅比较了单个模型的结果。最好性能模型是Deep Insight的，令人惊讶的是3个模型的集合仅提高了0.3个点，而GoogLeNet在模型集成时明显获得了更好的结果。

## 9.论文的贡献

### 9.1 设计了Inception模块

在多个不同尺寸的卷积核上同时进行卷积运算后再进行聚合，并使用1×1的卷积进行降维减少计算成本，从而将稀疏矩阵聚类成相对稠密子空间来实现对稀疏矩阵的优化。以往的网络结构如AlexNet和VGG都是通过增大网络的深度来提高性能，而Inception提供了另外一种思路，增强卷积模块功能，可以在增加网络深度和宽度的同时减少参数。加入了Inception模块后能加速网络并避免梯度爆炸。

### 9.2 使用了更小的卷积滤波器

采用1×1的卷积核有两个作用，第一是可以降低特征的维度，从而降低计算需求；第二是修正线性激活，在没有损失多少特征的基础上增加网络深度的同时增加了网络的宽度。

### 9.3 使用了更深的网络结构且设计了具有独创性的网络结构

作者在比赛中使用22层的深度神经网络GoogLeNet并取得了非常好的成绩。

> **GoogLeNet和VGG是2014年ImageNet竞赛的双雄，这两类模型结构有一个共同特点是go deeper。跟VGG不同的是，GoogLeNet做了更大胆的网络上的尝试而不是像VGG继承了LeNet以及AlexNet的一些框架，该模型虽然 有22层，但大小却比AlexNet和VGG都小很多，性能优越。**
>
> 参考：https://www.cnblogs.com/Allen-rg/p/5833919.html

### 9.4 使用了比AlexNet更为激进的图片修剪技术（即数据增强）

该文章将图像调整为256、288、320和352的尺寸的图像，取这些尺寸调整后的左中右三个部分的方格（如果是肖像画则取上中下三个部分的方格），对于每个方格取其左上、左下、右上、右下和中心的224 × 224大小的裁剪以及它们的镜像，总共每张图片有4 × 3 × 6 × 2 = 144个版本，相当于数据多了144倍。

> 参考：https://blog.csdn.net/yzqlyzql/article/details/90313882

### 9.5 使用辅助分类器

对深度相对较大的网络来说，梯度反向传播能够通过所有层的能力就会降低。辅助分类器相当于对模型做了融合，通过将辅助分类器添加到这些中间层，可以提高较低阶段分类器的判别力，这是在提供正则化的同时克服梯度消失问题。从中我们可以学习到一个点在于：梯度消失解决办法可以是在中间增加额外信号。

### 9.6 替换全连接层

将最后一个卷积层后的全连接层替换为池化层，减少了参数，节约了计算量。

### 9.7 最大贡献--Inception系列

本文提出的Inception即为Inception V1，在此基础上衍生出了一系列改进，有Inception V2、V3、V4及Inception-ResNet、Xception等，具体可参考https://www.cnblogs.com/vincent1997/archive/2019/05/24/10920036.html

## 10. 下一步应该/可以做什么?(本文的局限性)

未来的工作可以在本文的基础上以自动化的方式创建更稀疏更精细的结构以及将Inception架构的思考应用到其他领域。

而通过观察Inception V2、V3、V4对Inception V1的改进，可以发现其主要是通过引入归一化、卷积分解、与其他模型结合等方法来提高网络性能，提供了后续的研究思路。

## 11. 重要的相关论文

1. S. Arora, A. Bhaskara, R. Ge, and T. Ma. Provablebounds for learning some deep representations. CoRR,abs/1310.6343, 2013.
   
   在文中引用了6次，提出在特征维度上进行稀疏连接的方法，网上都说这篇论文对数学的要求很高，很难看懂。

2. D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov.Scalable object detection using deep neural networks.
   In CVPR, 2014.
   

在文中引用了5次，提出“DeepMultiBox”这种使用CNN来生成候选区域的方法。

3. R. B. Girshick, J. Donahue, T. Darrell, and J. Malik.Rich feature hierarchies for accurate object detection and semantic segmentation. In Computer Vision andPattern Recognition, 2014. CVPR 2014. IEEE Conference, 2014.
   
   在文中引用了6次，提出R-CNN。
   
4. M. Lin, Q. Chen, and S. Yan. Network in network. CoRR, abs/1312.4400, 2013.
   

在文中引用了7次，提出了1*1卷积及用全局均值池化层代替全连接层，改进了传统的CNN网络，采用了少量的参数就轻轻松松击败了Alexnet网络，影响深远，被ResNet 和 Inception 等网络模型借鉴。

5. Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[J]. arXiv preprint arXiv:1502.03167, 2015.

   提出Inception V2。

6. Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2818-2826.

   提出Inception V3。

7. Szegedy C, Ioffe S, Vanhoucke V, et al. Inception-v4, inception-resnet and the impact of residual connections on learning[C]//Thirty-First AAAI Conference on Artificial Intelligence. 2017.

   提出Inception V4。

8. Chollet F. Xception: Deep learning with depthwise separable convolutions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1251-1258.

   提出Xception。

## 12. 不懂之处

为什么引入稀疏性会造成计算能力浪费，

为什么inception结构可以在保持网络结构的稀疏性同时又能利用密集矩阵的高计算性能

## 13. 词汇

| 英文                                     | 中文               | 位置                                                      |
| ---------------------------------------- | ------------------ | --------------------------------------------------------- |
| embedded computing                       | 嵌入式计算         | 1. Introduction                                           |
| sparsity                                 | 稀疏性             | 3. Motivation and High Level Considerations               |
| sparse matrix/matrices                   | 稀疏矩阵           | 3. Motivation and High Level Considerations               |
| dense matrix/matrices                    | 密集矩阵           | 3. Motivation and High Level Considerations               |
| ensemble                                 | 组合               | 5. GoogLeNet                                              |
| reduction/projection layers              | 降维/投影层        | 5. GoogLeNet                                              |
| auxiliary classifier                     | 辅助分类器         | 5. GoogLeNet                                              |
| data-parallelism                         | 数据并行           | 6. Training Methodology                                   |
| asynchronous stochastic gradient descent | 异步随机梯度下降法 | 6. Training Methodology                                   |
| ground truth                             | 实际结果           | 7. ILSVRC 2014 Classification Challenge Setup and Results |
| bounding box                             | 边界框             | 8. ILSVRC 2014 Detection Challenge Setup and Results      |
| false positive                           | 假阳性             | 8. ILSVRC 2014 Detection Challenge Setup and Results      |
| mean average precision (mAP)             | 平均精度均值       | 8. ILSVRC 2014 Detection Challenge Setup and Results      |
| object bounding box recall               | 目标边界框召回率   | 8. ILSVRC 2014 Detection Challenge Setup and Results      |
| bounding box regression                  | 边界框回归         | 8. ILSVRC 2014 Detection Challenge Setup and Results      |
| contextual model                         | 上下文模型         | 8. ILSVRC 2014 Detection Challenge Setup and Results      |


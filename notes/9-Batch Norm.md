# 9-Batch Norm

## 1 文章想要解决的问题

通过在网络中加入Batch Norm层，消除了internal covariate shift，加快了训练速度。

### 1.1 internal covariate shift

**定义**：在深层网络训练的过程中，由于网络中参数变化而引起内部结点数据分布发生变化的这一过程被称作Internal Covariate Shift。

**理解**：定义网络每一层的线性变换为 ![[公式]](https://www.zhihu.com/equation?tex=Z%5E%7B%5Bl%5D%7D%3DW%5E%7B%5Bl%5D%7D%5Ctimes+input%2Bb%5E%7B%5Bl%5D%7D)，其中 ![[公式]](https://www.zhihu.com/equation?tex=l+) 代表层数；非线性变换为 ![[公式]](https://www.zhihu.com/equation?tex=A%5E%7B%5Bl%5D%7D%3Dg%5E%7B%5Bl%5D%7D%28Z%5E%7B%5Bl%5D%7D%29) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=g%5E%7B%5Bl%5D%7D%28%5Ccdot%29) 为第 ![[公式]](https://www.zhihu.com/equation?tex=l) 层的激活函数。

随着梯度下降的进行，每一层的参数 ![[公式]](https://www.zhihu.com/equation?tex=W%5E%7B%5Bl%5D%7D) 与 ![[公式]](https://www.zhihu.com/equation?tex=b%5E%7B%5Bl%5D%7D) 都会被更新，那么 ![[公式]](https://www.zhihu.com/equation?tex=Z%5E%7B%5Bl%5D%7D) 的分布也就发生了改变，进而 ![[公式]](https://www.zhihu.com/equation?tex=A%5E%7B%5Bl%5D%7D) 也同样出现分布的改变。而 ![[公式]](https://www.zhihu.com/equation?tex=A%5E%7B%5Bl%5D%7D) 作为第 ![[公式]](https://www.zhihu.com/equation?tex=l%2B1) 层的输入，意味着 ![[公式]](https://www.zhihu.com/equation?tex=l%2B1) 层就需要去不停适应这种数据分布的变化，这一过程就被叫做Internal Covariate Shift。

### 1.2 Internal Covariate Shift带来的问题

**（1）上层网络需要不停调整来适应输入数据分布的变化，导致网络学习速度的降低**

我们在上面提到了梯度下降的过程会让每一层的参数 ![[公式]](https://www.zhihu.com/equation?tex=W%5E%7B%5Bl%5D%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=b%5E%7B%5Bl%5D%7D) 发生变化，进而使得每一层的线性与非线性计算结果分布产生变化。后层网络就要不停地去适应这种分布变化，这个时候就会使得整个网络的学习速率过慢。

**（2）网络的训练过程容易陷入梯度饱和区，减缓网络收敛速度**

当我们在神经网络中采用饱和激活函数（saturated activation function）时，例如sigmoid，tanh激活函数，很容易使得模型训练陷入梯度饱和区（saturated regime）。随着模型训练的进行，我们的参数 ![[公式]](https://www.zhihu.com/equation?tex=W%5E%7B%5Bl%5D%7D) 会逐渐更新并变大，此时 ![[公式]](https://www.zhihu.com/equation?tex=Z%5E%7B%5Bl%5D%7D%3DW%5E%7B%5Bl%5D%7DA%5E%7B%5Bl-1%5D%7D%2Bb%5E%7B%5Bl%5D%7D) 就会随之变大，并且 ![[公式]](https://www.zhihu.com/equation?tex=Z%5E%7B%5Bl%5D%7D) 还受到更底层网络参数 ![[公式]](https://www.zhihu.com/equation?tex=W%5E%7B%5B1%5D%7D%2CW%5E%7B%5B2%5D%7D%2C%5Ccdots%2CW%5E%7B%5Bl-1%5D%7D) 的影响，随着网络层数的加深， ![[公式]](https://www.zhihu.com/equation?tex=Z%5E%7B%5Bl%5D%7D) 很容易陷入梯度饱和区，此时梯度会变得很小甚至接近于0，参数的更新速度就会减慢，进而就会放慢网络的收敛速度。

对于激活函数梯度饱和问题，有两种解决思路。第一种就是更为非饱和性激活函数，例如线性整流函数ReLU可以在一定程度上解决训练进入梯度饱和区的问题。另一种思路是，我们可以让激活函数的输入分布保持在一个稳定状态来尽可能避免它们陷入梯度饱和区，这也就是Normalization的思路。

## 2.研究的是否是一个新问题

ICS不是一个新问题，由于ICS产生的原因是因参数更新带来的网络中每一层输入值分布的改变，并且随着网络层数的加深而变得更加严重，因此我们可以通过固定每一层网络输入值的分布来对减缓ICS问题。

**（1）白化（Whitening）**

白化（Whitening）是机器学习里面常用的一种规范化数据分布的方法，主要是PCA白化与ZCA白化。白化是对输入数据分布进行变换，进而达到以下两个目的：

- **使得输入特征分布具有相同的均值与方差。**其中PCA白化保证了所有特征分布均值为0，方差为1；而ZCA白化则保证了所有特征分布均值为0，方差相同；
- **去除特征之间的相关性。**

通过白化操作，我们可以减缓ICS的问题，进而固定了每一层网络输入分布，加速网络训练过程的收敛（LeCun et al.,1998b；Wiesler&Ney,2011）。

**（2）Batch Normalization提出**

既然白化可以解决这个问题，为什么我们还要提出别的解决办法？当然是现有的方法具有一定的缺陷，白化主要有以下两个问题：

- **白化过程计算成本太高，**并且在每一轮训练中的每一层我们都需要做如此高成本计算的白化操作；
- **白化过程由于改变了网络每一层的分布**，因而改变了网络层中本身数据的表达能力。底层网络学习到的参数信息会被白化操作丢失掉。

既然有了上面两个问题，那我们的解决思路就很简单，一方面，我们提出的normalization方法要能够简化计算过程；另一方面又需要经过规范化处理后让数据尽可能保留原始的表达能力。于是就有了简化+改进版的白化——Batch Normalization。

## 3.提出的新知识

由于采用full batch的训练方式对内存要求较大，且每一轮训练时间过长；我们一般都会采用对数据做划分，用mini-batch对网络进行训练。因此，Batch Normalization也就在mini-batch的基础上进行计算。

![](https://upload-images.jianshu.io/upload_images/24408091-f7670d7f7b3e27aa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

对每个特征进行独立的normalization。考虑一个batch的训练，传入m个训练样本，并关注网络中的某一层，忽略上标 ![[公式]](https://www.zhihu.com/equation?tex=l) 。

![](https://upload-images.jianshu.io/upload_images/24408091-5d860cc4b952cef3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 是为了防止方差为0产生无效计算。

通过上面的变换，**我们解决了第一个问题，即用更加简化的方式来对数据进行规范化，使得第 ![[公式]](https://www.zhihu.com/equation?tex=l) 层的输入每个特征的分布均值为0，方差为1。**

如同上面提到的，Normalization操作我们虽然缓解了ICS问题，让每一层网络的输入数据分布都变得稳定，但却导致了数据表达能力的缺失。也就是我们通过变换操作改变了原有数据的信息表达（representation ability of the network），使得底层网络学习到的参数信息丢失。另一方面，通过让每一层的输入分布均值为0，方差为1，会使得输入在经过sigmoid或tanh激活函数时，容易陷入非线性激活函数的线性区域。

因此，BN又引入了两个可学习（learnable）的参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 与 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta) 。

![](https://upload-images.jianshu.io/upload_images/24408091-3811869a090a4eda.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这两个参数的引入是为了恢复数据本身的表达能力，对规范化后的数据进行线性变换。特别地，当 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma%5E2%3D%5Csigma%5E2%2C%5Cbeta%3D%5Cmu) 时，可以实现等价变换（identity transform）并且保留了原始输入特征的分布信息。

![](https://upload-images.jianshu.io/upload_images/24408091-459e3d39af0073cc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 4.关键的相关工作及相关人物

### 4.1 相关工作

#### 4.1.1 白化whitening

算法介绍和原理可参考博客：

[1]: https://blog.csdn.net/hjimce/article/details/50864602	"机器学习（七）白化whitening"

#### 4.1.2 Inception v1

实验部分的网络是基于v1版本基础上添加BN以及其他修改。

可参考其论文：

[2]: https://arxiv.org/pdf/1409.4842v1.pdf	"Going deeper with convolutions"



### 4.2 相关人物

#### 4.2.1 Sergey Ioffe

<img src="https://upload-images.jianshu.io/upload_images/24408091-be75d0544e88d314.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" style="zoom:67%;" />

#### 4.2.2 Christian Szegedy

![](https://upload-images.jianshu.io/upload_images/24408091-005a9f97aa14d0c1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 5.文章解决方法的关键

变换重构，引入了可学习参数$\gamma、\beta$, 让我们的网络可以学习恢复出原始网络所要学习的特征分布。

![](https://upload-images.jianshu.io/upload_images/24408091-ba6ed29c4609a872.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 6.实验设计

### 6.1 随时间激活

为了验证ICS对训练的影响，以及BN对抗它的能力，考虑了在MNIST数据集上预测数字类别的问题。使用非常简单的网络，28x28的二值图像作为输入，以及三个全连接层，每层100个激活。每一个隐藏层用sigmoid非线性计算$y=g(Wu+b)$，权重$W$初始化为小的随机高斯值。最后的隐藏层之后是具有10个激活（每类1个）和交叉熵损失的全连接层。我们训练网络50000次迭代，每份小批量数据中有60个样本。在网络的每一个隐藏层后添加BN。

![](https://upload-images.jianshu.io/upload_images/24408091-293b4bbe5ab8d7ca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

图(a)使用批标准化和不使用批标准化训练的网络在MNIST上的测试准确率，以及训练的迭代次数。批标准化有助于网络训练的更快，取得更高的准确率。(b，c)典型的sigmoid在训练过程中输入分布的演变，显示为15%，50%，85%。批标准化使分布更稳定并降低了内部协变量转移。

### 6.2 ImageNet分类

评估了几个带有批标准化的Inception修改版本。

#### 6.2.1 加速BN网络

将批标准化简单添加到网络中不能充分方法的优势。为此，进行了修改：

提高学习率。

删除Dropout。

更彻底地打乱训练样本。

减少L2正则化。

加速学习率衰减。

删除局部响应归一化。

降低亮度失真。

#### 6.2.2 单网络分类

评估了下面的网络，所有的网络都在LSVRC2012训练数据上训练，并在验证数据上测试：

*Inception*：以0.0015的初始学习率进行训练。

*BN-Baseline*：每个非线性之前加上批标准化，其它的与Inception一样。

*BN-x5*：带有批标准化的Inception，初始学习率增加5倍到了0.0075。原始Inception增加同样的学习率会使模型参数达到机器无限大。

*BN-x30*：类似于*BN-x5*，但初始学习率为0.045（Inception学习率的30倍）。

*BN-x5-Sigmoid*：类似于*BN-x5*，但使用sigmoud非线性g(t)=11+exp(−x)g(t)=11+exp⁡(−x)来代替ReLU。我们也尝试训练带有sigmoid的原始Inception，但模型保持在相当于机会的准确率。

![img](http://noahsnail.com/images/bn/Figure_2.png)

Inception和它的批标准化变种在单个裁剪图像上的验证准确率以及训练步骤的数量。

通过仅使用批标准化（BN-Baseline），在不到Inception一半的训练步骤数量内将准确度与其相匹配。显著提高了网络的训练速度。*BN-x5*需要比Inception少14倍的步骤就达到了72.2％的准确率。

#### 6.2.3组合分类

使用了6个网络。每个都是基于BN-x30的，进行了以下一些修改：增加卷积层中的初始重量；使用Dropout（丢弃概率为5％或10％，而原始Inception为40％）；模型最后的隐藏层使用非卷积批标准化。每个网络在大约6⋅1066⋅106个训练步骤之后实现了最大的准确率。组合预测是基于组成网络的预测类概率的算术平均。组合和多裁剪图像推断的细节与（Szegedy et al，2014）类似。

![](https://upload-images.jianshu.io/upload_images/24408091-7d26983f9c408a6e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上图证实了批标准化能够在ImageNet分类挑战基准上设置新的最佳结果。批标准化Inception与以前的最佳结果在提供的包含5万张图像的验证集上的比较。组合结果是在测试集上由测试服务器评估的结果。BN-Inception组合在验证集的5万张图像上取得了4.9% top-5的错误率。

## 7.使用的数据集

MNIST（开源）：由60000个训练样本和10000个测试样本组成，每个样本都是一张28 * 28像素的灰度手写数字图片。

LSVRC2012（开源）：ImageNet2012竞赛的数据集

## 9.论文的贡献

提出了一个新的机制，大大加快了深度网络的训练。具体如下：

**（1）BN使得网络中每层输入数据的分布相对稳定，加速模型学习速度**

BN通过规范化与线性变换使得每一层网络的输入数据的均值与方差都在一定范围内，使得后一层网络不必不断去适应底层网络中输入的变化，从而实现了网络中层与层之间的解耦，允许每一层进行独立学习，有利于提高整个神经网络的学习速度。

**（2）BN使得模型对网络中的参数不那么敏感，简化调参过程，使得网络学习更加稳定**

在神经网络中，我们经常会谨慎地采用一些权重初始化方法（例如Xavier）或者合适的学习率来保证网络稳定训练。

当学习率设置太高时，会使得参数更新步伐过大，容易出现震荡和不收敛。但是使用BN的网络将不会受到参数数值大小的影响。在使用Batch Normalization之后，抑制了参数微小变化随着网络层数加深被放大的问题，使得网络对参数大小的适应能力更强，此时我们可以设置较大的学习率而不用过于担心模型divergence的风险。

**（3）BN允许网络使用饱和性激活函数（例如sigmoid，tanh等），缓解梯度消失问题**

在不使用BN层的时候，由于网络的深度与复杂性，很容易使得底层网络变化累积到上层网络中，导致模型的训练很容易进入到激活函数的梯度饱和区；通过normalize操作可以让激活函数的输入数据落在梯度非饱和区，缓解梯度消失的问题；另外通过自适应学习 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 与 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta) 又让数据保留更多的原始信息。

**（4）BN具有一定的正则化效果**

在Batch Normalization中，由于我们使用mini-batch的均值与方差作为对整体训练样本均值与方差的估计，尽管每一个batch中的数据都是从总体样本中抽样得到，但不同mini-batch的均值与方差会有所不同，这就为网络的学习过程中增加了随机噪音，与Dropout通过关闭神经元给网络训练带来噪音类似，在一定程度上对模型起到了正则化的效果。

## 10.未来工作（论文的局限性）

### 10.1 BN缺点

- 对batch size的大小比较敏感，由于每次计算均值和方差是在一个batch上，所以如果batch size太小，则计算的均值、方差不足以代表整个数据分布；
- BN实际使用时需要计算并且保存某一层神经网络batch的均值和方差等统计信息，对于对一个固定深度的前向神经网络（DNN，CNN）使用BN，很方便；但对于RNN来说，sequence的长度是不一致的，换句话说RNN的深度不是固定的，不同的time-step需要保存不同的statics特征，可能存在一个特殊sequence比其他sequence长很多，这样training时，计算很麻烦。

### 10.2 未来工作

- 将BN方法应用于RNN
- 研究BN是否能够更容易泛化到新的数据分布
- 对BN的正则化属性进行更多的的研究
- 进一步理论分析得到更多的改进和应用

## 11. 重要的相关论文

Group Normalization:https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html

Layer Normalization:https://arxiv.org/abs/1607.06450

Weight Normalization:https://papers.nips.cc/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Paper.pdf



## 12. 不懂之处

一些涉及数学理论的地方没太懂，比如白化操作为什么计算代价高，不可微分

## 13.英文术语

| 英文                             | 中文释义       | 位置                                        |
| -------------------------------- | -------------- | ------------------------------------------- |
| regularizer                      | 正则化项       | Abstract                                    |
| internal covariate shift         | 内部协变量转移 | Abstract                                    |
| stochastic gradient descent(SGD) | 随机梯度下降法 | 1Introduction                               |
| convergence                      | 收敛           | 1 Introduction                              |
| differentiable                   | 可微的         | 3 Normalization via Mini-Batch Statistics   |
| element-wise                     | 逐元素地       | 3.2 Batch-Normalized Convolutional Networks |
| nonlinearity                     | 非线性         | 3.2 Batch-Normalized Convolutional Networks |
| benchmark                        | 基准           | 4.2.3 Ensemble Classification               |











# SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE

提出了SqueezeNet轻量级模型，将AlexNet压缩到了50倍，使得模型可以顺利地运行在硬件条件有限的嵌入式设备中。

### 1  论文的结构

#### 1.1  Abstract

- 作者想解决什么问题？

  如何在不降低准确率的同时压缩模型，在受限的硬件设备上运行神经网络模型

- 作者通过什么理论/模型来解决这个问题？

  SqueezeNet

- 作者给出的答案是什么？

  （1）在分布式训练中，与服务器通信需求更小；
  （2）参数更少，从云端下载模型的数据量小；
  （3）更适合在FPGA等内存受限的设备上部署。

#### 1.2 Introduction

- 作者为什么研究这个课题？

  近来深层卷积网络的主要研究方向集中在提高正确率。对于一个给定的精度水平，通常有多个CNN架构可以达到这个精度水平。对于相同的正确率水平，更小的CNN架构可以提供如下的优势：

  （1）更高效的分布式训练
  服务器间的通信是分布式CNN训练的重要限制因素。对于分布式数据并行训练方式，通信需求和模型参数数量正相关。小模型对通信需求更低。
  （2）减小下载模型到客户端的额外开销
  比如在自动驾驶中，经常需要更新客户端模型。更小的模型可以减少通信的额外开销，使得更新更加容易。
  （3）便于FPGA和嵌入式硬件上的部署

#### 1.3 Related work

##### 1.3.1 Model Compression

（1）奇异值分解(singular value decomposition (SVD))
（2）网络剪枝（Network Pruning）：使用网络剪枝和稀疏矩阵
（3）深度压缩（Deep compression）：使用网络剪枝，数字化和huffman编码
（4）硬件加速器（hardware accelerator）

##### 1.3.2 CNN Microarchitecture

在设计深度网络架构的过程中，通常先构建由几个卷积层组成的小模块，再将模块堆叠形成完整的网络，定义这种模块的网络为CNN Microarchitecture。CNN微结构相关工作主要有采用5×5、3×3、1×3 and3×1，即修改卷积核。

##### 1.3.3 CNN Macroarchitecture

将CNN Macroarchitecture定义为，将多个模块组织成一个端到端CNN的系统体系结构。介绍了VGG和ResNet，同时提到了本文引用了ResNet的旁路连接。

##### 1.3.4 Neural Network Design Space Exploration

由于超参数繁多，深度神经网络具有很大的**设计空间（design space）**。通常进行设计空间探索的方法有：
（1）贝叶斯优化
（2）模拟退火
（3）随机搜索
（4）遗传算法

但文章中指出以往工作在NN空间设计探索的不足：未能直观地看到NN设计空间。

#### 1.4  SqueezeNet: Prserving Accuracy with few Parameters

##### 1.4.1 Architectural Design Strategies

使用以下三个策略来减少SqueezeNet设计参数
（1）使用1x1卷积代替3x3 卷积：参数减少为原来的1/9
（2）减少输入通道数量：这一部分使用squeeze layers来实现
（3）将降采样操作延后，可以给卷积层提供更大的激活图：更大的激活图保留了更多的信息，可以提供更高的分类准确率
其中，（1）和（2）可以显著减少参数数量，（3）可以在参数数量受限的情况下提高准确率。

##### 1.4.2 THE Fire Module

Fire Module是SqueezeNet中的基础构建模块，如下定义 **Fire Module** :

<img src="https://img-blog.csdnimg.cn/20210222194347660.png" alt="在这里插入图片描述" style="zoom:80%;" />

SqueezeNet 设计思路有 3 个

1. squeeze convolution layer：只使用1x1卷积 filter，即以上提到的**策略（1）**
2. expand layer：使用1x1 和3x3卷积 filter的组合
3. Fire module中使用3个可调的超参数：s<sub>1x1</sub>（squeeze convolution layer中1×1 filter的个数）,e<sub>1x1</sub>（expand layer中1×1 filter的个数）和s<sub>3x3</sub>（expand layer中3×3 filter的个数）,
4. 使用Fire module的过程中，令s<sub>1x1</sub> < e<sub>1x1</sub> + s<sub>3x3</sub>，这样squeeze layer可以限制输入通道数量，即以上提到的**策略（2）**

##### 1.4.3 THE SqueezeNet Architecture

SqueezeNet以卷积层（conv1）开始，接着使用8个Fire modules (fire2-9)，最后以卷积层（conv10）结束。每个fire module中的filter数量逐渐增加，并且在conv1, fire4, fire8, 和 conv10这几层之后使用max-pooling，即将池化层放在相对靠后的位置，这使用了以上的**策略（3）**。

如下图，左边为原始的SqueezeNet，中间为包含simple bypass的改进版本，最右侧为使用complex bypass的改进版本。在下表中给出了更多的细节。

<img src="https://img-blog.csdnimg.cn/2021022220050458.png" alt="在这里插入图片描述" style="zoom:80%;" />

#### 1.5 Experiment

- 在哪些数据集下进行了测试

  Imagenet

- 实验中的重要指标有哪些？

  Top-1、Top-5

- 实验的设置过程中的trick
  （1）为了使 1×1 和 3×3 filter输出的结果具有相同的尺寸，在expand modules中，给3×3 filter的原始输入添加一个像素的边界（zero-padding）。
  （2）squeeze 和 expand layers中都是用ReLU作为激活函数
  （3）在fire9 module之后，使用Dropout，比例取50%
  （4）SqueezeNet中没有全连接层，这借鉴了Network in network的思想
  （5）训练过程中，初始学习率设置为0.04，在训练过程中线性降低学习率。
  （6）由于Caffe中不支持使用两个不同尺寸的filter，在expand layer中实际上是使用了两个单独的卷积层（1×1 filter 和 3×3 filter），最后将这两层的输出连接在一起，这在数值上等价于使用单层但是包含两个不同尺寸的filter。
  在github上还有SqueezeNet在其他框架下的实现：MXNet、Chainer、Keras、Torch。

#### 1.6 Conclusion

提出了SqueezeNet模型，参数比AlexNet减少了50倍，并在imagenet上保持AlexNet水平的精度。

### 2 相关的关键人物与工作

#### 2.1 Song Han

<img src="https://songhan.mit.edu/wp-content/uploads/2020/02/songhan_square-1024x1024.jpg" alt="img" style="zoom:50%;" />

Song Han是麻省理工学院EECS的助理教授。他在斯坦福大学获得博士学位，研究方向是efficient deep learning computing. 

<img src="https://www.hualigs.cn/image/60345fe735994.jpg"/>

#### 2.2 W.Dally

<img src="https://www.hualigs.cn/image/60346134868ca.jpg"/>

### 3 论文提出的解决方案的关键

使用以下三个策略来减少SqueezeNet设计参数
（1）使用1x1卷积代替3x3 卷积：参数减少为原来的1/9
（2）减少输入通道数量：这一部分使用squeeze layers来实现
（3）将降采样操作延后，可以给卷积层提供更大的激活图：更大的激活图保留了更多的信息，可以提供更高的分类准确率
其中，（1）和（2）可以显著减少参数数量，（3）可以在参数数量受限的情况下提高准确率。

### 4 实验支撑

#### 4.1 论文的数据集哪里获取

Imagenet（http://www.image-net.org/）

#### 4.2 源代码哪里可以获取

https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py

#### 4.3 关键代码的讲解

##### 4.3.1 Fire Module

```python
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
```

首先Fire类是对`torch.nn.Modules`类进行拓展得到的，需要继承Modules类，并实现`__init__()`方法，以及`forward()`方法。其中，`__init__()`方法用于定义一些新的属性，这些属性可以包括Modules的实例，如一个`torch.nn.Conv2d`，`nn.ReLU`等。即创建该网络中的子网络，在创建这些子网络时，这些网络的参数也被初始化。接着使用`super(Fire, self).__init__()`调用基类初始化函数对基类进行初始化。

在Fire类的`__init__()`函数中，定义了如下几个新增的属性：

* `inplanes`：输入向量
* `squeeze：sqeeze layer`，由二维1×1卷积组成。其中，参考PyTorch文档，`torch.nn.Conv2d` 的定义为`class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`，代码中inplanes为输入通道，squeeze_planes为输出通道，卷积模板尺寸为1×1.
* `expand1x1`：expand layer中的1×1卷积。
* `expand3x3`：expand layer中的3×3卷积。为了使 1×1 和 3×3 filter输出的结果有相同的尺寸，在expand modules中，给3×3filter的原始输入添加一个像素的边界（zero-padding）
* 所有的激活函数都选择ReLU。`inplace=True`参数可以在原始的输入上直接进行操作，不会再为输出分配额外的内存，可以节省一部分内存，但同时也会破坏原始的输入。

##### 4.3.2 SqueezeNet

```python
class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)
```
SqueezeNet类同样继承自nn.Module，这里实现了version=1.0和version=1.1两个SqueezeNet版本。区别在于1.0只有 AlexNet的1/50的参数，而1.1在1.0的基础上进一步压缩，参数略微减少，计算量降低为1.0的40%左右。SqueezeNet类定义了如下属性：

1. `num_classes`：分类的类别个数
2. `self.features`：定义了主要的网络层，`nn.Sequential()`是PyTorch中的序列容器（sequential container），可以按照顺序将Modules添加到其中。
3. `nn.MaxPool2d()`函数中`ceil_mode=True`会对池化结果进行向上取整。
4. 最后一个卷积层的初始化方法与其他层不同，在接下来的for循环中，定义了不同层的初始化方法，最后一层使用了均值为0，方差为0.01的正太分布初始化方法，其余层使用**He Kaiming**论文中的均匀分布初始化方法。同时这里使用了`num_classes`参数，可以调整分类类别数目。并且所有的bias初始化为零。
5. `self.classifier`：定义了网络末尾的分类器模块，注意其中使用了`nn.Dropout()`
6.  `forward()`方法：可以看到由features模块和classifier模块构成。

### 5 实验结果

##### 5.1 Evaluation of SqueezeNet

SqueezeNet 有 8 个 Fire Module 模块，而每个模块有 3 个超参数，所以它总共有 24 个超参数。

<img src="https://img-blog.csdnimg.cn/20210222210808173.png" alt="在这里插入图片描述" style="zoom:80%;" />

**对比了不同的模型压缩方法。**

1、SVD方法能将预训练的AlexNet模型压缩为原先的1/5，Top-1正确率略微降低。

2、网络剪枝的方法能将模型压缩到原来的1/9，Top-1和Top-5正确率几乎保持不变。

3、深度压缩能将模型压缩到原先的1/35，正确率基本不变。

4、SqeezeNet的压缩倍率可以达到50以上，并且正确率还能有略微的提升。如果将深度压缩（Deep Compression）的方法用在SqeezeNet上，使用33%的稀疏表示和8位精度，会得到一个仅有0.66MB的模型。进一步，如果使用6位精度，会得到仅有0.47MB的模型，同时正确率不变。
此外，结果表明深度压缩不仅对包含庞大参数数量的CNN网络作用，对于较小的网络，比如SqueezeNet，也是有用的。将SqueezeNet的网络架构创新和深度压缩结合起来可以将原模型压缩到1/510。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222203119674.png)

##### 5.2 CNN Microarchitecture Design Space Exploration

下面的左图给出了压缩比（SR）对模型精度和大小的影响，右图给出3×3卷积占比对模型精度和大小的影响。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222203218899.png)

##### 5.3 Tradinng off  1X1 and 3X3 Filters

受ResNet启发，这里探究旁路连接（bypass conection）的影响。使用旁路连接后正确率确实有一定提高。下表给出了实验结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210222203326635.png)

### 6 论文最大的贡献

​	提出了三个策略来压缩模型，在不降低准确率的情况下降低了模型的参数量，最大的贡献是开拓了模型压缩这一方向，之后的一系列文章也就此打开。

### 7 论文的不足之处

#### 7.1 经典轻量化模型

#####  7.2.1 MobileNet v1

采用名为Depth-wise seperable convolution的卷积代替传统的卷积计算方式，以达到减少网络权值参数的目的。MobileNet将传统的标准卷积分解为两步计算：Depth-wise convolution和Pointwise convolution。 Depth-wise convolution是特殊的分组卷积，即分组数等于卷积通道数，即逐通道卷积，一个卷积核负责一个通道； Pointwise convolution就是1x1 Convolution，将Depth-wise convolution的feature map串起来，以解决“信息不流通的问题”。采用Depth-wise convolution相较于传统卷积，可成倍地减少计算量，从而达到减少模型参数量和提升运算速度的目的； 采用Pointwise convolution解决Depth-wise convolution的“信息不流通”问题。

##### 7.2.2 ShuffleNet v1

利用group convolution和channel shuffle代替传统convolution，以减少参数量。采用group convolution会导致“信息流通不畅”的问题，因此提出channel shuffle。channels shuffle是将group convolution得到的feature map的channels进行有序打乱，够成新的feature map，以解决“信息流通”问题。 对比MobileNet，ShuffleNet采用channel shuffle代替1x1 conv可有效减少参数和计算量。

![在这里插入图片描述](https://www.pianshen.com/images/492/7ff2a510df5026ebbad8bd7b7631f4ec.png)

(PS：Group convolution是将输入层的不同特征图进行分组，然后采用不同的卷积核再对各个组进行卷积，这样会降低卷积的计算量。因为一般的卷积都是在所有的输入特征图上做卷积，即全通道卷积，这是一种通道密集连接方式（channel dense connection），而group convolution相比则是一种通道稀疏连接方式)

#### 7.2 本文的缺点

1. SqueezeNet的侧重的应用方向是嵌入式环境，目前嵌入式环境主要问题是实时性。SqueezeNet通过更深的深度置换更少的参数数量虽然能减少网络的参数，但是其丧失了网络的并行能力，测试时间反而会更长，SqueezeNet的Fire模块的两个分支的计算方式不同，在GPU并行计算两个分支时，运算量较小的分支会等待运算量较大的分支，于是丧失了网络的并行性，因为小分支的计算量小的优点无法体现出来。
2. SqueezeNet得到的模型是5MB左右，0.5MB是经过Deep Compression进行模型压缩后的结果。

#### 7.3改进方向

* 针对卷积操作进行优化

  例如GhostNet: More Features from Cheap Operations（CVPR 2020 华为诺亚实验室）

  mobilenet v2

  [![yLVueH.png](https://s3.ax1x.com/2021/02/23/yLVueH.png)](https://imgchr.com/i/yLVueH)


### 8 重要的相关论文

1. E.L Denton, W. Zaremba, J. Bruna, Y. LeCun, and R. Fergus. Exploiting linear structure within
   convolutional networks for efficient evaluation. In NIPS, 2014.

   提出奇异值分解(singular value decomposition (SVD))

2. S. Han, J. Pool, J. Tran, and W. Dally. Learning both weights and connections for efficient neural
   networks. In NIPS, 2015b.

   提出网络剪枝（Network Pruning），使用网络剪枝和稀疏矩阵。

3. **S. Han, H. Mao, and W. Dally. Deep compression: Compressing DNNs with pruning, trained**
   **quantization and huffman coding. arxiv:1510.00149v3, 2015a.** 

   **提出了深度压缩（Deep compression），使用网络剪枝，数字化和huffman编码。**


### 9 相关资料

https://blog.csdn.net/csdnldp/article/details/78648543

https://zhuanlan.zhihu.com/p/49465950

https://frank909.blog.csdn.net/article/details/84995553

https://blog.csdn.net/u013044310/article/details/80188530

https://blog.csdn.net/qq_31914683/article/details/79513874

https://zhuanlan.zhihu.com/p/90274900
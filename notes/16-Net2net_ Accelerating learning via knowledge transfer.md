## 16-Net2net_ Accelerating learning via knowledge transfer

Net2Net(Net to Net) 是利用知识迁移来解决**大型网络的训练速度慢**的问题。

### 1  论文的结构(简要概括)

- 作者想解决什么问题？

​        加速大型神经网络训练速度

- 作者通过什么理论/模型来解决这个问题？

​        functionpreserving transformations

- 作者给出的答案是什么？

​       可以通过知识迁移加速训练

#### 1.2 Introduction

- 作者为什么研究这个课题？

​       传统的机器学习算法是接受一个固定的数据集作为输入，在不接受任何知识情况下初始化，并训练模型至收敛。但实际应用场景中，数据集往往是不断增长的，为避免过拟合和降低模型计算成本，一开始会选择小模型，之后需要一个大模型以充分利用大型数据集。而重新训练一个大的神经网络十分耗时。

#### 1.4  Methodology

- 作者是用什么理论证明了自己的方法在理论上也是有保障的？

​       functionpreserving transformations

#### 1.5 Experiments

- 作者是在哪些数据集或者说场景下进行了测试？

​      ImageNet

- 实验中的重要指标有哪些？

​      Training Accuracy，Validation Accuracy随迭代次数的变化

#### 1.6 Discussion

- 这篇论文最大的贡献是什么？

​       提出Net2Net方法，在现有模型基础上加速新模型训练

- 论文中的方法还存在什么问题？

​       不可泛化

- 作者觉得还可以怎么改进？

​       可设计更多通用的知识转移的方法让其可以快速初始化一个学生网络，并且网络的结构不受限于教师网络。

#### 1.6 Related work

- 和作者这篇论文相关的工作有哪些？

​        级联相关、模型压缩、深度置信网络DBN

- 作者主要是对之前的哪个工作进行改进？

​       Inception+BN+RMSProp作为基准Net2Net

### 2 论文想要解决的问题？

解决**大型网络的训练速度慢**的问题。

例如先训练一个小的网络，然后Net2Net，训练一个更大的网络，训练更大的网络时可以利用在小网络中已经训练好的权重，使得再训练大型的网络速度就变的非常快，利用小网络的权重的这个过程就是知识迁移的过程。

#### 2.1 背景是什么？

传统的机器学习算法是接受一个固定的数据集作为输入，在不接受任何知识情况下初始化，并训练模型至收敛。

#### 2.2 之前的方法存在哪些问题

每个模型从头训练浪费时间。

### 3 论文研究的是否是一个新问题

可能是？似乎没有看到跟本文目的完全相同的

以往的迁移学习：预训练后加层的时候改变了原本的function

本文：从先前训练好的模型中提取知识，作为新网络的初始化。

### 4 论文试图验证的科学假设

使用Net2Net操作初始化的模型比标准模型收敛得更快

### 5 相关的关键人物与工作

#### 5.1 相关的关键人物

##### 5.1.1 Tianqi Chen（陈天奇）

陈天奇是机器学习领域著名的青年华人学者之一，本科毕业于上海交通大学ACM班，博士毕业于华盛顿大学计算机系，研究方向为大规模机器学习。

**主要贡献**：

创建了ML系统

- [Apache TVM](https://tvm.apache.org/)，一个用于深度学习的自动化端到端优化编译器。
- [XGBoost](https://xgboost.ai/)，可扩展的树增强系统。
- [Apache MXNet](https://mxnet.incubator.apache.org/)（共同创建者）

**代表性论文**：

<img src="https://upload-images.jianshu.io/upload_images/24408091-ead922e792e44a48.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" style="zoom:67%;" />

##### 5.1.2 Ian Goodfellow

目前受雇于Apple Inc.作为Special Projects Group的机器学习主管，以前是Google Brain的研究科学家，2014年蒙特利尔大学机器学习博士。
他的研究兴趣涵盖大多数深度学习主题，特别是生成模型以及机器学习的安全和隐私。在研究对抗样本方面是一位有影响力的早期研究者，他发明了生成式对抗网络，在深度学习领域贡献卓越。

**主要贡献**：

- 提出了生成对抗网络（GANs），被誉为“GANs之父”
- 《深度学习》花书作者之一

**代表性论文**：

<img src="https://upload-images.jianshu.io/upload_images/24408091-213ee1b53b97cbca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" style="zoom:67%;" />

##### 5.1.3 Jonathon Shlens

Google Brain的资深研究员，致力于机器学习，计算机视觉和计算神经科学研究。

**主要贡献**：

- TensorFlow的共同发明者

**代表性论文**：

<img src="C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20201206094611321.png" alt="image-20201206094611321" style="zoom:67%;" />

#### 5.2 相关的关键工作

##### 5.2.1 文章的baseline

Inception+BN+RMSProp

##### 5.2.2 级联相关（Cascade-correlation)

级联相关神经网络是从一个小网络开始，自动训练和添加隐含单元，最终形成一个多层的结构。可视为Net2Net系列一部分。

**缺点**：在新组建添加到网络后到其仍然未适应之前，会经历一段时间低性能。

**本文方法**：保留功能的初始化（function-preserving initializations）避免了这段低性能。

##### 5.2.3 模型压缩（model compression)

一种将知识从许多模型转移到单个模型的技术。 它的目的不同于我们的Net2Net技术。 

**目的**: 通过使最终模型学习与许多不同模型学习的平均函数相似的函数，来规范最终模型。 

**本文目的**： Net2Net旨在通过利用现有模型中的知识来非常快速地训练最终模型，但是与从头开始训练最终模型相比，它没有尝试使最终模型更加规范化。

##### 5.2.4 深度置信网络（DBN）

本文方法与DBN区别：保留功能的转换（function-preserving transformations ）在意义上更为笼统，它们允许添加一个宽度大于其下一层的宽度的图层，而DBN增长仅针对一种特定大小的新层保留分布。

### 6 论文提出的解决方案的关键

神经网络的参数之间function-preserving transformations，使得网络的拓扑结构改变后还能利用旧网络的权重，改变拓扑结构但是没有改变网络的效果，对于同样的输入有同样的输出。

### 7 论文的解决方案有完备的理论证明吗

有，文中理论证明：

提出了一种新的大型神经网络操作方法（Net2Net）：将一个神经网络中的知识快速转移到另一个神经网络中。

![](https://upload-images.jianshu.io/upload_images/24408091-f1a00cb76e8f0923.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

提出了两种特定的Net2Net方法，Net2WiderNet和Net2DeeperNet。两者都基于神经网络的function-preserving transformations思想。

#### 7.1 Net2Net策略

##### 7.1.1 Feature prediction

从随机初始化开始训练一个大型的学生网络，并在学生网络中引入一组额外的教师预测层。学生被训练使用每一个隐藏层来预测老师隐藏层的值。

该方法没有提供明显优势，文章转向FUNCTION-PRESERVING INITIALIZATIONS。

##### 7.1.2 Function-preserving initializations

基于初始化学生网络来表示与教师相同的功能，然后通过正常的方式继续训练学生网络。

假设教师网络由一个函数$y=f(x ; \theta)$表示，$x$是网络的输入，$y$是网络的输出，$θ$是网络的参数。

文章的策略是选择一组新的参数$θ^{\prime}$表示学生网络$g(x ; \theta^{\prime})$，并且有
$$
\forall x, f(x ; \theta)=g\left(x ; \theta^{\prime}\right)
$$

#### 7.2 Net2Net方法

##### 7.2.1 Net2WiderNet

**直观理解**

允许一个层被一个更宽的层替换，这意味着一个层有更多的节点。对于卷积结构，则意味着层将有更多的卷积通道。

![](https://upload-images.jianshu.io/upload_images/24408091-c7107d8445b8102e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如上，对于一个全连接层来说，如果我们新增了一个节点，那么我们随机从已有节点中选择一个节点copy它的输入权重，使得这个节点的值和已有选择的节点的值相同，对于输出的节点来说，需要把前一层的节点的值求和激活，这时我们发现我们选择的那个节点的值扩大了两倍，于是我们可以把他们各自都除以2，这样我们就实现了全连接层的恒等替换。

对于一个卷积层来说，道理也类似，如果我想增加一个channel，我可以随机选一个channel然后copy它的权重(filter)，对于输出时要再进行卷积的filter而言，我们把filter中这两层的channel的权重除以2就可以，这样也在channel增加的情况实现了恒等替换。

**简单案例说明**

假设第$i$层和第$i + 1$层都是完全连接的层,第$i$层使用元素非线性。为扩展第$i$层，替换了$W (i)$和$W (i+1)$。如果第$i$层有$m$个输入和$n$个输出，第$i+1$层有$p$个输出，即$\boldsymbol{W}^{(i)} \in \mathbb{R}^{m \times n}$ ， $\boldsymbol{W}^{(i+1)} \in \mathbb{R}^{n \times p}$，Net2WiderNet允许我们替换有$q$个输出的第$i$层。

引入随机映射函数$g:\{1,2, \cdots, q\} \rightarrow\{1,2, \cdots, n\}$
$$
g(j)=\left\{\begin{array}{ll}
j & j \leq n \\
\text { random sample from }\{1,2, \cdots n\} & j>n
\end{array}\right.
$$
引入新的权值矩阵$U (i)$和$U (i+1)$表示新学生网络中这些层的权值。然后给出新的权重
$$
\boldsymbol{U}_{k, j}^{(i)}=\boldsymbol{W}_{k, g(j)}^{(i)}, \quad \boldsymbol{U}_{j, h}^{(i+1)}=\frac{1}{|\{x \mid g(x)=g(j)\}|} \boldsymbol{W}_{g(j), h}^{(i+1)}
$$
这里，$W (i)$的前$n$列被直接复制到$U (i)$中，第$n+1$行到$U(i)$通过选择一个$g$中定义的随机变量来创建。对于$U (i+1)$中的权值，将权值除以$\frac{1}{|\{x \mid g(x)=g(j)\}|}$，因此所有的单元都具有与原始网络中的单元完全相同的值。

![](https://upload-images.jianshu.io/upload_images/24408091-8a23995f2d03765b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



##### 7.2.2 Net2DeeperNet

![](https://upload-images.jianshu.io/upload_images/24408091-e9ab14af9a38cbed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

对于一个全连接层来说，我们利用一个单位矩阵做权值，添加一个和上一个全连接层维度完全相同的全连接层，把前一个全连接层的值copy过来，实现恒等映射，此时再结合Net2WiderNet，就可以使这一层再变宽。但是我们可能很难实现把这个层变瘦。

对于一个卷积层来说，我们利用一个只有中间位置是1，其它位置全是0的filter，就很轻松的实现了恒等映射。

### 8 实验设计

#### 8.1用到了哪些数据集

ImageNet

#### 8.2与什么算法进行了比较

所有案例中，都使用了**InceptionBN**网络(Ioffe &amp;Szegedy, 2015)

##### 8.2.1  Net2WiderNet

**Baseline**:“随机pad”。通过添加具有随机权重的新单元来扩展网络，而不是通过复制单元来执行保留功能的初始化。此操作是通过使用附加的随机值填充现有的权重矩阵来实现的。

从构建一个比标准初始更窄的教师网络开始。我们将inception model中每一层的卷积通道数量减少了一倍。由于减少了通道的输入和输出数量，这将大多数层中的参数数量减少到原始数量的30%。为了简化我们实验的软件，我们没有修改除在inception model之外的网络组件。在对这个小型教师网络进行训练之后，我们用它来加速对一个标准规模学生网络的训练。

##### 8.2.2  Net2DeeperNet

**Baseline**:“随机初始化”。将初始化加速的深度网络的训练与随机初始化的相同深度网络的训练进行了比较。

使用了一个标准的Inception模型作为教师网络，并增加了每个Inception模块的深度。Inception模块中的卷积层使用矩形内核。卷积层成对排列，一层使用垂直内核，另一层使用水平内核。每一层都是具有校正非线性和批量归一化的完整层；它不仅仅是将线性卷积运算分解成可分离的部分。在出现垂直-水平卷积层的任何地方，我们都添加了另外两对这样的层，它们被配置成一个恒等变换。

##### 8.2.3 Exploring model design space with Net2Net

Net2Net的一个重要特性是，通过转换现有的最先进的体系结构，可以快速探索模型空间。为对模型设计空间进行了更广泛和更深入的探索，文章将Inception模型的宽度扩大到原来模型的p2倍。我们还通过在初始inception模型的每个inception模块上添加四个垂直-水平卷积层对，构建了另一个更深层次的网络。

#### 8.3评价指标是什么

Training Accuracy，Validation Accuracy随迭代次数的变化

### 9 实验支撑

#### 9.1 论文的数据集哪里获取

ImageNet官网：http://image-net.org/index

#### 9.2 源代码哪里可以获取

未找到

#### 9.3 关键代码的讲解

github上的一个实现代码：https://github.com/DanielSlater/Net2Net

https://github.com/paengs/Net2Net

### 10 实验结果是否验证了科学假设？

#### 10.1  Net2WiderNet

![](https://upload-images.jianshu.io/upload_images/24408091-4323eca51551c43d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

比较不同方法，可以发现，该方法比基线方法具有更快的收敛速度。Net2WiderNet提供了与随机初始化训练的模型相同的最终精度。这说明模型的真实大小决定了训练过程的精度。初始化模型以模拟更小的模型不会损失准确性。因此，Net2WiderNet可以安全地用于更快地达到相同的精度，从而减少运行新实验所需的时间。

#### 10.2  Net2DeeperNet

![](https://upload-images.jianshu.io/upload_images/24408091-474d5fb870b703dc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

由结果发现Net2DeeperNet在训练和验证精度上都比随机初始化训练快得多。

#### 10.3 Exploring model design space with Net2Net

![](https://upload-images.jianshu.io/upload_images/24408091-90cdf09e79e92ce1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

由结果可以发现，使用Net2Net操作初始化的模型比标准模型收敛得更快。

### 11 论文最大的贡献

1. 提出保留功能的初始化策略，有如下优点：

   - 新的大网络和原来的性能一样，不花费时间在之前低性能时期训练；
   - 保证在初始化后的任何更改都是改进的，之前的方法可能无法在baseline上改进，因为对较大模型初始化后的更改恶化了性能
   - 对网络中所有参数的优化都是“安全的”，从来没有哪个阶段某一层会接收到有害的梯度、需要冻结。这与级联相关（cascade correlation）等方法形成对比，后者将冻结旧单元，以避免在试图影响新的随机连接单元的行为时产生不良的适应性。

2. 提出Net2Net方法，在现有模型基础上加速新模型训练

3. 应用于终身学习系统：

   真实场景下的机器学习系统，最终都会变成**终身学习系统**(Lifelong learning system)，不断的有新数据，通过新的数据改善模型，刚开始数据量小，我们使用小的网络，可以防止过拟合并加快训练速度，但是随着数据量的增大，小网络就不足以完成复杂的问题了，这个时候我们就需要在小网络上进行扩展变成一个大网络了。

   Net2Net操作使我们能够顺利地实例化一个大得多的模型，并立即开始在我们的终身学习系统中使用它，而不需要花费数周或数月的时间在最新的、最大版本的训练集上从头开始重新训练一个大模型。

### 12 论文的不足之处

文章通过定义两个操作，实现了从小（教师）网络到大（学生）网络的转换，因为不是所有的情况都有事先恒等映射的方法，故这个方法不可泛化

#### 12.1 这篇论文之后的工作有哪些其他的改进

论文中说未来可设计更多通用的知识转移的方法让其可以快速初始化一个学生网络，并且网络的结构不受限于教师网络。

#### 12.2 你觉得可以对这篇论文有什么改进

设计一个普适的算法，可以将任意深度宽度的小网络的权重迁移到任意深度宽度的大网络中。

### 13 重要的相关论文

实验的baseline和框架：

Inception

>  Szegedy, Christian, Liu, Wei, Jia, Yangqing, Sermanet, Pierre, Reed, Scott, Anguelov, Dragomir, Erhan, Dumitru,
>  Vanhoucke, Vincent, and Rabinovich, Andrew. Going deeper with convolutions. Technical report,
>  arXiv:1409.4842, 2014.

BN

> Ioffe, Sergey and Szegedy, Christian. Batch normalization: Accelerating deep network training by reducing
> internal covariate shift. 2015.

RMSProp

> Tieleman, T and Hinton, G. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent
> magnitude. COURSERA: Neural Networks for Machine Learning, 4, 2012.

Tensorflow

> Abadi, Mart´ın, Agarwal, Ashish, Barham, Paul, Brevdo, Eugene, Chen, Zhifeng, Citro, Craig, Corrado, Greg S.,
> Davis, Andy, Dean, Jeffrey, Devin, Matthieu, Ghemawat, Sanjay, Goodfellow, Ian, Harp, Andrew, Irving,
> Geoffrey, Isard, Michael, Jia, Yangqing, Jozefowicz, Rafal, Kaiser, Lukasz, Kudlur, Manjunath, Leven-berg, Josh, Man´e, Dan, Monga, Rajat, Moore, Sherry, Murray, Derek, Olah, Chris, Schuster, Mike, Shlens,
> Jonathon, Steiner, Benoit, Sutskever, Ilya, Talwar, Kunal, Tucker, Paul, Vanhoucke, Vincent, Vasudevan,
> Vijay, Vi´egas, Fernanda, Vinyals, Oriol, Warden, Pete, Wattenberg, Martin, Wicke, Martin, Yu, Yuan, and
> Zheng, Xiaoqiang. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. URL
> http://tensorflow.org/. Software available from tensorflow.org.

### 14 不懂之处

文章在Net2WiderNet中随机选择单元复制后加了一点噪声，说是为了打破对称，对称有什么不好的影响吗？而对于Net2DeeperNet却说不需要加噪声。
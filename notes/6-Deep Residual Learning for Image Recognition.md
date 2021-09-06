## 6 Deep Residual Learning for Image Recognition

### 1 文章想要解决的问题

本文希望解决超深度CNN网络训练问题。

对于梯度消失/爆炸问题：它从一开始就阻碍了收敛，但很大程度上可以通过标准初始化和正则化来解决（SGD+反向传播）

而面对的退化问题：网络不是越深越好。随着网络深度的增加，精度会饱和，然后迅速退化，且并不是由过拟合引起的。

![img](https://upload-images.jianshu.io/upload_images/145616-fb67b4fbdb0670c3.png?imageMogr2/auto-orient/strip|imageView2/2/w/729/format/webp)

### 2 研究的是否是一个新问题

不算是，之前此类问题已经提出，没有被很好的解决。但是理论上说，如果添加层只是一个恒等映射(identity mapping)，即指添加的额外层数没有起效果，等效于直接将结果输出。理论上网络越深不会导致训练误差越高。所以我们认为可能存在另一种构建方法，随着深度的增加，训练误差不会增加，只是我们没有找到该方法而已。因此本文提出了Residual Networks(ResNet)，2015 ImageNet中分类任务的冠军。

### 3 本文采用的新知识

提出残差学习的动机：源于退化问题。如果添加的层可以构造为恒等映射(identity mapping)，那么较深的模型的训练误差应该不大于较浅的模型。

通过残差学习来重构模型，进行预处理，如果恒等映射是最优的，求解器可以简单地将多个非线性层的权值趋近于零来逼近恒等映射。

#### 3.1 普通网络设计（参考VGG）

1. 对于相同的尺寸的输出特征图谱，每层必须含有相同数量的过滤器。

2. 如果特征图谱的尺寸减半，则过滤器的数量必须翻倍，以保持每层的时间复杂度。

3. 卷积层（stride=2）进行下采样，网络末端以全局的均值池化层结束，1000的全连接层（Softmax激活）。共34层。

   这样构建出的34层网络计算量远小于VGG-19

#### 3.2 残差网络设计
基于普通网络，其中插入快捷连接(shortcut)来实现残差学习。快捷连接不引入额外的参数和复杂度。不仅在实践中很有吸引力，而且方便比较普通网络和残差网络，具有相同数量的参数、深度、宽度和计算成本。

![img](https://img-blog.csdnimg.cn/2019022417482452.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

### 4 相关的关键人物与工作

#### 4.1 相关的关键人物

何凯明博士，2007年清华大学毕业之后开始在微软亚洲研究院（MSRA）实习，2011年香港中文大学博士毕业后正式加入MSRA，目前在Facebook AI Research (FAIR)实验室担任研究科学家。曾以第一作者身份拿过两次CVPR最佳论文奖（2009和2016）——其中2016年CVPR最佳论文为图像识别中的深度残差学习（Deep Residual Learning for Image Recognition），深入浅出地描述了深度残差学习框架，大幅降低了训练更深层次神经网络的难度，也使准确率得到显著提升。

#### 4.2 残差表示

图为残差网络的一个结构块。不是让网络直接拟合原先的映射，而是拟合残差映射。比如用H(X)来表示最优解映射，但是让堆叠的非线性层去拟合另一个映射F(X) = H(X) - X, 原始映射被重新转换为F(x) + X。

实际上，把残差推至0和把此映射逼近另一个非线性层相比要容易的多。许多论文已经证明使用残差的解法收敛速度比不用残差的普通解法要快的多。这些研究表明，一个好的模型重构或者预处理手段是能简化优化过程的。

![img](https://upload-images.jianshu.io/upload_images/145616-89fa3a044f73b1d1.png?imageMogr2/auto-orient/strip|imageView2/2/w/639/format/webp)

#### 4.3 快捷连接

F(x) + x 的公式可以通过具有快捷连接的前馈神经网络来实现 ，快捷连接跳过一个或多个层。且不增加参数和复杂性。

其中，针对虚线的快捷连接尺寸变化处理，有三种选择：

（A）zero-padding shortcuts。快捷连接仍然使用恒等映射，对不足的特征维数直接进行补零。

（B）projection shortcuts。利用1x1卷积核进行升降维匹配尺寸。其他的shortcut都是恒等映射（identity mapping）类型。

（C）所有的shortcuts都使用projection shortcuts。

#### 4.4 恒等（identity) vs 投影(projection)捷径

Identity数学式：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbold+y+%3D+%5Cmathcal%7BF%7D%28%5Cbold+x%2C%5C%7BW_i%5C%7D%29+%2B+%5Cbold+x.)

x和y分别表示构造块的输入和输出向量，函数$\mathcal{F}\left(\mathbf{x},\left\{W_{i}\right\}\right)$表示被将被训练的残差映射。

比如对图2的结构块，$F=W_{2} \sigma\left(W_{1} \mathrm{x}\right)$，σ表示ReLU。F + x 是由一个快捷连接进行逐元素的添加得,做加法后得到的模型具有二阶非线性。

projection数学式：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbold+y+%3D+%5Cmathcal%7BF%7D%28%5Cbold+x%2C%5C%7BW_i%5C%7D%29+%2B+W_s%5Cbold+x.)

在快捷连接上进行一个线性投影$W_s$来匹配维度。论文中projection 的实现方式是通过一个 1*1的卷积核来匹配维度。

#### 4.5 深度瓶颈架构

![image.png](https://upload-images.jianshu.io/upload_images/24435792-f17fdad99a18c1f9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


减少训练时间，改为一种瓶颈设计。对于每个残差函数F，将2层结构改为3层，先用1x1压缩通道数目以减少计算量，3x3进行卷积，再用1x1进行升维，带来更高的效率。而图左因为该快捷连接连到了两个高维端，耗费太多时间。

### 5 文章提出的解决方案的关键

残差网络ResNet，除了使用残差外，点睛之笔在于作者通过快捷连接来实现了一个简单的恒等映射。如下图所示。

![img](https://pic1.zhimg.com/80/v2-5b80976f389fd0d379e42e92eeecbf44_720w.jpg)

作者通过优化 ![[公式]](https://www.zhihu.com/equation?tex=F%28x%29) ，使得 ![[公式]](https://www.zhihu.com/equation?tex=F%28x%29%5Crightarrow0) ，从而使得 ![[公式]](https://www.zhihu.com/equation?tex=x%5Crightarrow+F%28x%29%2Bx%5Crightarrow+x) ，最终达到一个恒等映射的关系。最终的实验表明，引入残差学习后，深层的网络更容易优化并且不会产生更高的训练错误率，甚至还能降低错误率。


### 6 实验设计
利用不同的数据集验证模型泛化能力。

对同一个数据集：

1. 对比残差网络和普通网络，看退化问题是不是得到解决了。收敛速度和响应强度区别。
2. 相同的残差网络，比较不同的shortcuts策略。
3. ResNet和其它先进算法，对比效果。
4. 探索了残差网络层数问题；100层和1000层ResNet的效果对比。

#### 6.1 ImageNet 分类数据集

基于ImageNet，图片被根据短边等比缩放，按照[256,480]区间的尺寸随机采样进行尺度增强。裁切224x224随机抽样的图像或其水平翻转，并将裁剪结果减去它的平均像素值，标准颜色的增强。批量正则化用在了每个卷积层和激活层之间，初始化权重为0。

使用SGD算法，mini-batch=256，$\alpha_0$=0.1，（当到达迭代要求时把学习速率除以10），权重衰减，衰减率=0.0001，动量=0.9，无dropout。

1000个类的ImageNet 2012分类数据集。训练集：128万。测试集：50k。

##### 6.1.1 普通网络vs残差网络

![img](https://upload-images.jianshu.io/upload_images/145616-de468a4dab9c8ad2.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

![img](https://upload-images.jianshu.io/upload_images/145616-fe0ee692f2b90797.png?imageMogr2/auto-orient/strip|imageView2/2/w/711/format/webp)

Imagenet的训练，细的曲线表示训练集误差，粗的曲线表示验证集误差。左：普通网络（18/34层），右：残差网络（18/34层）。所有的短连接使用恒等映射，对增加的维度使用零填充(选项A)。

可以看出普通网络的退化问题，随着网络深度的增加，误差率反而上升。原因尚不明了，深的普通网络可能指数级的降低的收敛速度，对训练误差的降低产生影响。

结果对比：
1.消除退化问题，并且可推广到验证数据。
2.对比普通网络，34层ResNet将错误降低了3.5，验证了在深度网络上残差学习的有效性。
3.收敛更快，ResNet在早期提供更快的收敛速度。

##### 6.1.2 恒等捷径(identity) vs 投影(projection)捷径

针对三种shortcuts实验来对比。

![image.png](https://upload-images.jianshu.io/upload_images/24435792-93a82492b59795f3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

表3，C>B>A，但C牺牲了太多训练时间。考虑采用B方案+深度瓶颈架构加快训练时间。152层的ResNet比VGG16/19 复杂度少得多。

##### 6.1.3 和其它先进技术比较

![image.png](https://upload-images.jianshu.io/upload_images/24435792-9ca5775c652e9537.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

与其他最先进技术的比较。34层已经很好。深度增加后精度也增加，152层top-5的验证误差率达到4.49%。结果优于所有以前的综合模型。其中比赛时ResNet152在测试集上获得了3.57% 的top-5误差。这在2015 ILSVRC获得了第一名。



####  6.2 CIFAR-10数据集

衰减率=0.0001动量=0.9，Batch-normalization，无dropout。两个gpu，mini-batch=128。$\alpha_0$=0.1，在32k和48k次迭代时$\alpha$除以10，在64k次迭代时终止训练。数据增强策略：在各边缘增加4像素，32x32的切割完全随机，从填充后的图像或者其翻转中采样。

CIFAR-10数据集：训练集：50K；测试集：10K；分10类。
架构：输入32x32的图像，预先减去每一个像素的均值。第一层是3×3卷积层。对于尺寸分别为{32, 16, 8 }的特征图谱分别使用过滤器{16,32,64}，降采样为步长为2的卷积，网络以全局的均值池化终止，10全连通层，softmax。

![img](https://upload-images.jianshu.io/upload_images/145616-cbb1f94ccb1b783e.png?imageMogr2/auto-orient/strip|imageView2/2/w/620/format/webp)

采用A策略恒等映射，使残差模型跟普通模型有这一模一样的深度、宽度和参数个数。结果网络对数据集依然有很好的效果。

![image.png](https://upload-images.jianshu.io/upload_images/24435792-44acfb64ee2469c6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 6.2.1 网络层响应分析

![image.png](https://upload-images.jianshu.io/upload_images/24435792-7999dca939d1e86b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

图7展示了残差函数的响应强度，残差函数通常比非残差函数更接近于零，更深的ResNet则具有更小的响应量。当有更多的层时，单个层的ResNets倾向于较少地修改信号。

##### 6.2.2 1000层以上

1000层没有显示出优化的困难，误差率增加可能是过拟合。

![img](https://upload-images.jianshu.io/upload_images/145616-b3b061e96e3cc417.png?imageMogr2/auto-orient/strip|imageView2/2/w/797/format/webp)

#### 6.3 其它数据集（PASCAL和MS COCO）

对其它数据集也有强泛化能力：PASCAL VOC 2007、2012、COCO的目标检测基线，COCO 2015竞赛:ImageNet检测、ImageNet定位、COCO检测、COCO分割等获得多个第一。

### 7 采用的数据集及其是否开源

[code](https://github.com/KaimingHe/deep-residual-networks)

[何凯明ppt讲解](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxkZWVwbm50ZW1wfGd4OjZhNjExMDVhODQ5MWJjMzQ)

### 8 实验结果是否验证了科学假设

利用第6节的实验，可以看出残差网络明显解决网络退化问题，使得引入残差学习后，深层的网络更容易优化并且不会产生更高的训练错误率，甚至还能降低错误率。

### 9. 本文贡献

提出Residual Networks(ResNet)，一个残差学习的框架，以减轻网络的训练负担，比以往的网络要深的多。其具有以下优点：

1. 残差网络很容易习得映射关系，因而比普通深层网络更容易训练。
2. 随着深度的增加，精度也会增加，效果良好。对于CIFAR-10数据集，超过100层的网络表现很成功，还可以扩展到1000层。
3. ImageNet对象分类数据集比赛，152层的残差网络参赛网络中最深的，然而却拥有比VGG更低的复杂度。2015 ImageNet比赛中以3.57%的误差率，获得第一名。
4. 在其他识别任务中也有良好的泛化能力，多个比赛的第一名（有ImageNet detection, Imagenet localization,COCOdetection COCOsegmentation），证明残差学习的原则是可泛化的。

### 10 下一步怎么做（本文局限性）

退化问题的原因尚不明了，深的普通网络可能指数级的降低的收敛速度，训练误差的降低产生影响。

本文重点讲述ResNet对退化问题的处理，而对如何实施了正规化来配合增大/减少网络深度架构设计，结合更强的正规化手段进一步提升结果讲述甚少。下一步可以向此方向探索。

### 11 重要的相关论文

[1]同期的快捷连接：[Highway Networks](http://arxiv.org/pdf/1505.00387v2.pdf) 

[2]. [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 

[3]. [Batch Normalization](http://arxiv.org/pdf/1502.03167v3.pdf) 

[4]. [VGG](http://arxiv.org/pdf/1409.1556.pdf)

### 12.  不懂之处：残差网络解决了什么，为什么有效？

残差网络在图像领域已然成为了一种主流模型，虽然这种网络范式的提出是为了解决网络退化问题，但是关于其作用的机制，还是多有争议。目前存在几种可能的解释，下面分别列举2016年的两篇文献和2018年的一篇文献中的内容。

#### 12.1 从前后向信息传播的角度来看

何恺明等人从前后向信息传播的角度给出了残差网路的一种解释[[5\]](https://zhuanlan.zhihu.com/p/80226180#ref_3)。

考虑残差块![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D%5E%7B%28l%29%7D%3D%5Cmathcal%7BH%7D%28%5Cmathbf%7Ba%7D%5E%7B%28l-1%29%7D%29%3D%5Cmathbf%7Ba%7D%5E%7B%28l-1%29%7D%2B%5Cmathcal%7BF%7D%28%5Cmathbf%7Ba%7D%5E%7B%28l-1%29%7D%29%5Ctag%7B5%7D)

组成的前馈神经网络，为了讨论简便，暂且假设残差块不使用任何激活函数，即

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Ba%7D%5E%7B%28l%29%7D%3D%5Cmathbf%7Bz%7D%5E%7B%28l%29%7D%5Ctag%7B6%7D)

考虑任意两个层数 ![[公式]](https://www.zhihu.com/equation?tex=l_2%3El_1) ，递归地展开（5）（6）

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D%5Cmathbf%7Ba%7D%5E%7B%28l_2%29%7D%26%3D%5Cmathbf%7Ba%7D%5E%7B%28l_2-1%29%7D%2B%5Cmathcal%7BF%7D%28%5Cmathbf%7Ba%7D%5E%7B%28l_2-1%29%7D%29%5C%5C+%26%3D%5Cleft%28%5Cmathbf%7Ba%7D%5E%7B%28l_2-2%29%7D%2B%5Cmathcal%7BF%7D%28%5Cmathbf%7Ba%7D%5E%7B%28l_2-2%29%7D%29%5Cright%29%2B%5Cmathcal%7BF%7D%28%5Cmathbf%7Ba%7D%5E%7B%28l_2-1%29%7D%29%5C%5C+%26%3D%5Ccdots+%5Cend%7Balign%7D%5Ctag%7B7%7D)

可以得到

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Ba%7D%5E%7B%28l_2%29%7D%3D%5Cmathbf%7Ba%7D%5E%7B%28l_1%29%7D%2B%5Csum_%7Bi%3Dl_1%7D%5E%7Bl_2-1%7D%5Cmathcal%7BF%7D%28%5Cmathbf%7Ba%7D%5E%7B%28i%29%7D%29%5Ctag%7B8%7D)

根据式 （8） ，在前向传播时，**输入信号可以从任意低层直接传播到高层**。由于包含了一个天然的恒等映射，**一定程度上可以解决网络退化问题**。

这样，最终的损失 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 对某低层输出的梯度可以展开为

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+%5Cepsilon%7D%7B%5Cpartial+%5Cmathbf%7Ba%7D%5E%7B%28l_1%29%7D%7D%3D%5Cfrac%7B%5Cpartial+%5Cepsilon%7D%7B%5Cpartial+%5Cmathbf%7Ba%7D%5E%7B%28l_2%29%7D%7D%2B%5Cfrac%7B%5Cpartial+%5Cepsilon%7D%7B%5Cpartial+%5Cmathbf%7Ba%7D%5E%7B%28l_2%29%7D%7D%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%5Cmathbf%7Ba%7D%5E%7B%28l_1%29%7D%7D%5Csum_%7Bi%3Dl_1%7D%5E%7Bl_2-1%7D%5Cmathcal%7BF%7D%28%5Cmathbf%7Ba%7D%5E%7B%28i%29%7D%29%5Ctag%7B10%7D)

根据式（10） ，损失对某低层输出的梯度，被分解为了两项，前一项 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+%5Cepsilon%7D%7B%5Cpartial+%5Cmathbf%7Ba%7D%5E%7B%28l_2%29%7D%7D) 表明，反向传播时，**错误信号可以不经过任何中间权重矩阵变换直接传播到低层，一定程度上可以缓解梯度弥散问题（即便中间层矩阵权重很小，梯度也基本不会消失）**。

综上，可以认为**残差连接使得信息前后向传播更加顺畅。**

\* 加入了激活函数的情况的讨论(实验论证)请参见[[5\]](https://zhuanlan.zhihu.com/p/80226180#ref_3)。

#### 12.2 集成学习的角度

Andreas Veit等人提出了一种不同的视角[[6\]](https://zhuanlan.zhihu.com/p/80226180#ref_2)。他们将残差网络展开，以一个三层的ResNet为例，将得到下面的树形结构：

![img](https://pic2.zhimg.com/80/v2-2be86e9a6c5165054f770c395fc48219_720w.jpg)残差网络的展开形式

使用图来表示就是

![img](https://pic2.zhimg.com/80/v2-0fc8ceade0ecea1f87a8230575f23db9_720w.jpg)残差网络的展开形式

这样，**残差网络就可以被**看作是**一系列路径集合组装而成的一个集成模型**，其中不同的路径包含了不同的网络层子集。Andreas Veit等人展开了几组实验（Lesion study），在测试时，删去残差网络的部分网络层（即丢弃一部分路径）、或交换某些网络模块的顺序（改变网络的结构，丢弃一部分路径的同时引入新路径）。实验结果表明，网络的表现与正确网络路径数平滑相关（在路径变化时，网络表现没有剧烈变化），**这表明残差网络展开后的路径具有一定的独立性和冗余性，使得残差网络表现得像一个集成模型（ensemble）。**

作者还通过实验表明，残差网络中主要在**训练中贡献了梯度的是那些相对较短的路径**，从这个意味上来说，残差网络并不是通过保留整个网络深度上的梯度流动来抑制梯度弥散问题，一定程度上反驳了何恺明等[[5\]](https://zhuanlan.zhihu.com/p/80226180#ref_3)中的观点。但是，**我觉得这个实验结果与何凯明等的结论并不矛盾，因为这些较短的梯度路径正是由残差结构引入的**。

\* 可以类比集成学习的网络架构方法不仅有残差网络，Dropout机制也可以被认为是隐式地训练了一个组合的模型。

#### 12.3 梯度破碎问题

2018年的一篇论文，The Shattered Gradients Problem: If resnets are the answer, then what is the question?[[7\]](https://zhuanlan.zhihu.com/p/80226180#ref_4)，指出了一个新的观点，尽管残差网络提出是为了解决梯度弥散和网络退化的问题，**它解决的实际上是梯度破碎问题**(the shattering gradient problem):

> **在标准前馈神经网络中，随着深度增加，梯度逐渐呈现为白噪声(white noise)**。

作者通过可视化的小型实验(构建和训练一个神经网络 ![[公式]](https://www.zhihu.com/equation?tex=f%3A%5Cmathbb%7BR%7D%5Crightarrow+%5Cmathbb%7BR%7D) )发现，在浅层神经网络中，梯度呈现为棕色噪声(brown noise)，深层神经网络的梯度呈现为白噪声。在标准前馈神经网络中，随着深度增加，**神经元梯度的相关性(corelation)按指数级减少** ( ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B2%5EL%7D) )；同时，**梯度的空间结构也随着深度增加被逐渐消除**。这也就是梯度破碎现象。

![img](https://pic3.zhimg.com/80/v2-208275f71855e68b2fadaeac6b263d0a_720w.jpg)神经网络梯度及其协方差矩阵的可视化，可以看到标准的前馈网络的梯度在较深时(b)与白噪声(e)类似。

梯度破碎为什么是一个问题呢？这是因为**许多优化方法假设梯度在相邻点上是相似的，破碎的梯度会大大减小这类优化方法的有效性**。另外，如果梯度表现得像白噪声，那么某个神经元对网络输出的影响将会很不稳定。

相较标准前馈网络，**残差网络中梯度相关性减少的速度从指数级下降到亚线性级**(sublinearly, ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B%5Csqrt%7BL%7D%7D) )，深度残差网络中，神经元梯度介于棕色噪声与白噪声之间(参见上图中的c,d,e)；残差连接可以**极大地保留梯度的空间结构**。残差结构缓解了梯度破碎问题。

\* 更细致的实验与讨论请参见[[7\]](https://zhuanlan.zhihu.com/p/80226180#ref_4)。

### 13 专业术语

因为这一篇论文主要关于残差网络，提出的新知识/专业术语都在第3、4节有详细解释，此处不再赘述。

### 14 英语词汇

| 英文                            | 中文/解释     |
| ------------------------------- | ------------- |
| vanishing/exploding gradients   | 梯度消失/发散 |
| shortcut connections            | 快捷连接      |
| identity mappings               | 恒等映射      |
| degradation                     | 网络训练退化  |
| projection mappings             | 投影映射      |
| Deeper Bottleneck Architectures | 深度瓶颈架构  |
| .Residual Learning              | 残差学习      |




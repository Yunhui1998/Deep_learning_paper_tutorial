## 15 Improving neural networks by preventing co-adaptation of feature detectors

论文作者Hinton、Alex；2012年6月；

第一次提出Dropout技术，尝试解决一个过大的前馈神经网络应用到一个小的数据集上时对测试数据表现不佳的过拟合问题。

另一篇Dropout14年（第11篇）。

### 1  论文的结构

本篇论文的结构与常见的论文不一样，没有提供标题的方式划分章节，只有四个部分：摘要、正文、引用、实验。

#### 1.1  Abstract

- 作者想解决什么问题？
  过拟合问题：当一个大的前馈神经网络在一个小的训练集上训练时，它通常对测试数据表现不佳。
- 作者通过什么理论/模型来解决这个问题？
  Dropout：作者提出一种随机失活算法，这个算法通过减少神经元间的共适性来达到解决过拟合的目的。

#### 1.2 正文

- 为什么会出现这个问题？
  参数过多时会有很多权重都参与一个特征的检测以实现在训练集的完美检测，但是在测试集上这些权重会得出不一致、错误的结果，因此导致了过拟合。
- 作者使用的理论是基于哪些假设？
  通过平均大量不同的网络的结果，可以很好减少错误率。
- 目前这个课题的研究进行到了哪一阶段？存在哪些缺陷？作者是想通过本文解决哪个问题？
  文中没有提到其他解决过拟合的办法，而且实验也只是对比了Dropout使用的有无，没有与其他算法对比。
- 作者还给出了哪些解释？
  类比性别在生育中的作用，从进化的角度解释为什么Dropout是可行、有效的。

#### 1.5 Experiment

- 作者是在哪些数据集或者说场景下进行了测试？
  mnist手写字数据集、TIMIT语音语料库、路透社语料库、CIFAR-10微图像集、ImageNet。
  
- 实验中的重要指标有哪些？
  分类错误率Classification error。
  
- 在实验的设置过程中作者有没有提到自己用到了什么trick？
  在MNIST部分参数的设置如下：
  $$
  \epsilon_{0}=10.0, f=0.998, p_{i}=0.5, p_{f}=0.99, T=500
  $$
  

### 2 论文想要解决的问题？

#### 2.1 背景是什么？

过拟合问题：当一个大的前馈神经网络在一个小的训练集上训练时，它通常对测试数据表现不佳。

作者提出：参数过多时会有很多权重都参与一个特征的检测以实现在训练集的完美检测，但是在测试集上这些权重会得出不一致、错误的结果，因此导致了过拟合。

同时，提出过拟合问题是由于网络中存在大量的具有共适性的单元。共适性(co-adaptation)在文中没有详细解释，个人理解：多个神经元并不是相互独立的（类似非正交），每个神经元都无法单独检测某种特征，只有一起使用时才有检测能力；由于是多个神经元一起检测某种特征，因此很容易就会过拟合这个特征；这种“局部”过拟合或许可以理解为；当整个神经网络中充满了这种具有共适性的神经元后，整个网络也就过拟合了。

#### 2.2 之前的方法存在哪些问题

论文中没有对比之前的方法。

#### 2.3 输入和输出是什么？

Dropout是应用在神经网络中的一种算法，通过训练时随机失活，应用/测试时使用平均值而达到解决过拟合的效果。

### 3 论文研究的是否是一个新问题

过拟合已存在许久，之前也尝试过用诸如L2正则化、L1正则化等方式去解决这个问题。

### 4 论文试图验证的科学假设

通过训练时随机失活，应用时取平均值而达到如下效果：只使用较小的算力，就模拟出使用多个不同神经网络进行预测后取平均值一样的效果。

### 5 相关的关键人物与工作

#### 5.1 之前存在哪些相关的工作

大部分在吴恩达的课程中已经进行学习，已学过的只是简单提一下。

##### 5.1.1 Early Stopping

过拟合情况大致如下，那为了防止网络性能变差，可以选择在谷点停止训练，从而避免过拟合。

![过拟合](https://img-blog.csdn.net/20151026205032330)

但是缺点也显而易见，网络的性能很可能远没有达到最优，这样会造成一定浪费。

##### 5.1.2 数据增强

通过各种数字图像处理手段，增加图像的数量、去噪（过拟合会把噪声也拟合了）、增加噪声等方法来解决过拟合。

缺点是单纯地增加数据集可能无效，而且如果使用的处理方法不当会让网络学习到错误的知识（比如对文字进行镜像处理）。

##### 5.1.3 正则化

主要指L1正则化和L2正则化。通过在目标函数或代价函数后面加上一个正则项实现。

**L1正则化**：

cost函数：
$$
J(w, b)=\frac{1}{m} \sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y^{(i)}\right)+\frac{\lambda}{2 m}\|w\|_{1}
$$
更新公式：
$$
\begin{array}{c}
w:=w-\alpha d w\\
=w-\frac{\alpha \lambda}{2 m} \operatorname{sign}\left(w\right)-\frac{\partial J}{\partial w}
\end{array}
$$
sign为取符号函数，输入为正数返回1，负数返回-1，其余返回0。

若 w 为正数，则每次更新减去一个常数；若 w 为负数，则每次更新加上一个常数，故容易产生特征的系数为0的情况，特征系数为 0 表示该特征不会对结果有任何影响，因此L1正则化让w变得稀疏（更多的0），起到**特征选择**的作用。

**L2正则化**：

cost函数：
$$
J(w, b)=\frac{1}{m} \sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y^{(i)}\right)+\frac{\lambda}{2 m}\|w\|_{2}^{2}
$$
更新公式：
$$
w:=w-\alpha d w=\left(1-\frac{\alpha \lambda}{m}\right) w-\frac{\partial J}{\partial w}
$$
每次更新对特征系数做一个**比例的缩放**而不是像L1正则化减去一个固定值。这使得系数趋向变小而不会变为0，因此L2正则化**让模型变得更简单，防止过拟合**。

#### 5.2 本文是对哪个工作进行的改进

本文提出了一个全新的算法Dropout，可以说是使用最广泛、效果也相当好的算法。

其中比较小的一个改进是使用upper bound on the L2 norm（上界L2正则化），而不是通常的L2。

对比没有使用Dropout的网络，使用了Dropout的神经元看起来识别目标会更清晰，并且数值比较集中（越黑权值越大），由于识别的是手写数字而看起来更像是在检测特定的笔画。

![视觉效果](https://upload-images.jianshu.io/upload_images/16793245-c616f964005c9786.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 5.3 这个领域的关键研究者

本篇论文和第11篇论文（也是研究Dropout的，但更晚、引用更多）的作者完全一致，原班人马。

##### 5.3.1  R. Salakhutdinov

影响最大的四篇论文如下：

![image-20210127171741237.png](https://upload-images.jianshu.io/upload_images/16793245-2682a8c1996416f5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 6 论文提出的解决方案的关键

本文的重点在于首次提出Dropout技术。并且给出了一个比较重要的结论：多个不同网络的平均预测结果比任意一个独立的网络都更好。这看起来好像很天经地义，但就是这个思想造就了Dropout的成功。

用随机失活模拟不同的网络，用权值放缩模拟多个网络的平均预测，只是这样简单的方式就实现了刚刚的思想。

本文提出的Dropout的使用规则是：

1. 用mini-batch的随机梯度下降进行训练

2. 加入有上界的L2正则化，防止权重过大；这样就可以采用较大的 lr

3. 训练时，每个隐藏单元都有概率$1-p$激活（不作处理），概率$p$失活（变为0）。文中$p$为0.5

4. 权重更新方式：
   $$
   \begin{aligned}
   \Delta w^{t} &=p^{t} \Delta w^{t-1}-\left(1-p^{t}\right) \epsilon^{t}\left\langle\nabla_{w} L\right\rangle \\
   w^{t} &=w^{t-1}+\Delta w^{t}
   \end{aligned}
   $$
   其中
   $$
   \begin{aligned}
   \epsilon^{t} &=\epsilon_{0} f^{t} \\
   p^{t} &=\left\{\begin{array}{ll}
   \frac{t}{T} p_{i}+\left(1-\frac{t}{T}\right) p_{f} & t<T \\
   p_{f} & t \geq T
   \end{array}\right.
   \end{aligned}
   $$
   
5. 测试时，采用平均网络（mean network），即所有权重乘以0.5，变为原来的一半

假设一个网络有$N$个隐藏单元，则加入了Dropout的情况下理论上可等效为$2^N$个网络（每个单元有2种可能性）。

关于为什么要取平均值，我的理解是：在训练时，由于每个神经元都有$p$的概率失活，也就是说每个神经元的期望值是$w(1-p)$；但是在应用时，不再使用随机失活，期望值变为$w$；为了让训练和应用时的期望值一致，所以需要乘以$1-p$，也就是0.5。

在具体实现时，可以选择在训练时先除以$1-p$，这样就保持了训练时期望值为$w(1-p)/(1-p)=w$，自然实际应用时就无需任何额外操作。

### 7 论文的解决方案有完备的理论证明吗

没有提供比较完备的理论证明，个人认为只给出了一句比较能说明有效性的话：多个不同网络的平均预测结果比任意一个独立的网络都更好。A good way to reduce the error on the test set is to average the predictions produced by a very large number of different networks. 

### 8 实验设计

#### 8.1用到了哪些数据集

**MNIST数据集**：很常用的手写数字数据集。

**Tiny Image数据集**： 80 million张从网上收集的32 *×* 32有色图片。 

**CIFAR-10数据集**：Tiny Image数据集的子集，60000张10个类。

**ImageNet**：有大量图片的数据集，1000个类别。

**路透社语料库**（Reuters Corpus Volume I (RCV1-v2)）：一个包含804,414个新闻专线故事的存档，这些故事被手工分类为103个主题。

**TIMIT语料集**（TIMIT Acoustic--Phonetic Continuous Speech Corpus）：用于评价自动语音识别系统的标准数据集。

#### 8.2与什么算法进行了比较

没有与其他算法进行比较，空白对照。

#### 8.3评价指标是什么

**错误率**：错误样本数 / 总样本数

#### 8.4有没有什么独特的实验实验设计？

除了应用到神经网络的训练中，还将Dropout应用到了第4篇论文（Reducing the dimensionality of data with neural networks）中提到的方法（使用RBM初始化autoencoder）中的微调步骤。

除此之外，还可以发现这篇论文的附录部分相当长，描述了大量的实验的细节。

### 9 实验支撑

#### 9.1 论文的数据集哪里获取

**MNIST数据集**：Python的tf、keras等库都可以提供下载和使用。

**Tiny Image**：貌似已被下架，下架原因是有学者研究指出，这个通过大量搜索引擎整合的数据集，内里竟然隐藏着诸多令人不齿的**标签**：儿童猥亵、性暗示、种族歧视……（引自[这个比肩ImageNet的数据集遭MIT紧急下架，原因令人愤怒](https://zhuanlan.zhihu.com/p/153273238)）

网址：http://tiny-imagenet.herokuapp.com/

**CIFAR数据集**：网址http://www.cs.toronto.edu/~kriz/cifar.html.

**ImageNet数据集**：下载需要申请，不可商用，详细下载方式见[此篇](https://zhuanlan.zhihu.com/p/42696535)。

**路透社语料集**：网址http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm

**TIMIT语料集**：网址https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3/tech&hit=1&filelist=1

#### 9.2 源代码哪里可以获取

论文没有给出源码，只给出了实验中使用的其他源码。

层级的贪婪对比散度学习（greedy layer-wise Contrastive Divergence learning）：http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html

预训练深度玻尔兹曼机（ pretrained Deep Boltzmann Machine）：http://www.utstat.toronto.edu/~rsalakhu/DBM.html

Kaldi（用于语音的开源代码库）：http://kaldi.sourceforge.net

从网上找到的源码（其实就是tf中实现Dropout的源码）链接[在这](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py)。在函数dropout_v2。

```python
def dropout_v2(x, rate, noise_shape=None, seed=None, name=None):
  """
  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
    name: A name for this operation (optional).
  Returns:
    A Tensor of the same shape of `x`.
  """
  with ops.name_scope(name, "dropout", [x]) as name:
    is_rate_number = isinstance(rate, numbers.Real)
    if is_rate_number and (rate < 0 or rate >= 1):
      raise ValueError("rate must be a scalar tensor or a float in the "
                       "range [0, 1), got %g" % rate)
    x = ops.convert_to_tensor(x, name="x")
    x_dtype = x.dtype
    if not x_dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going "
                       "to be scaled. Got a %s tensor instead." % x_dtype)
    if is_rate_number and rate == 0:
      # Fast-path: Return the input immediately if rate is non-tensor & is `0`.
      # We trigger this after all error checking
      # and after `x` has been converted to a tensor, to prevent inconsistent
      # tensor conversions/error raising if rate is changed to/from 0.
      #
      # We also explicitly call `random_seed.get_seed` to make sure
      # we don't change the random number generation behavior of
      # stateful random ops by entering a fastpath,
      # despite not generating a random tensor in the fastpath
      random_seed.get_seed(seed)
      return x

    is_executing_eagerly = context.executing_eagerly()
    if not tensor_util.is_tf_type(rate):
      if is_rate_number:
        keep_prob = 1 - rate
        scale = 1 / keep_prob
        scale = ops.convert_to_tensor(scale, dtype=x_dtype)
        ret = gen_math_ops.mul(x, scale)
      else:
        raise ValueError("rate is neither scalar nor scalar tensor %r" % rate)
    else:
      rate.get_shape().assert_has_rank(0)
      rate_dtype = rate.dtype
      if rate_dtype != x_dtype:
        if not rate_dtype.is_compatible_with(x_dtype):
          raise ValueError(
              "Tensor dtype %s is incomptaible with Tensor dtype %s: %r" %
              (x_dtype.name, rate_dtype.name, rate))
        rate = gen_math_ops.cast(rate, x_dtype, name="rate")
      one_tensor = constant_op.constant(1, dtype=x_dtype)
      ret = gen_math_ops.real_div(x, gen_math_ops.sub(one_tensor, rate))

    noise_shape = _get_noise_shape(x, noise_shape)
    # Sample a uniform distribution on [0.0, 1.0) and select values larger
    # than rate.
    #
    # NOTE: Random uniform can only generate 2^23 floats on [1.0, 2.0)
    # and subtract 1.0.
    random_tensor = random_ops.random_uniform(
        noise_shape, seed=seed, dtype=x_dtype)
    # NOTE: if (1.0 + rate) - 1 is equal to rate, then that float is selected,
    # hence a >= comparison is used.
    keep_mask = random_tensor >= rate
    ret = gen_math_ops.mul(ret, gen_math_ops.cast(keep_mask, x_dtype))
    if not is_executing_eagerly:
      ret.set_shape(x.get_shape())
    return ret
```

### 10 实验结果是否验证了科学假设？

从空白对照来看，Dropout确实很明显地解决了过拟合问题。但是在本篇论文中没有验证“随机失活可以模拟多个不同的神经网络的结合”。

### 11 论文最大的贡献

Dropout确确实实为解决过拟合提出了很好的解决方案。

### 12 论文的不足之处

没有和其他防止过拟合的算法进行对比。

证明了Dropout的效果，但是没有很好地证明Dropout可以拟模拟多个不同的神经网络的结合。

#### 12.1 这篇论文之后的工作有哪些其他的改进

Dropout虽然已经很好了，但过拟合仍然是一个比较常见的问题。Dropout利用随机失活模拟多个不同的神经网络的结合，而且相当高效。如果想要进行改进，可能就得用其他思路防止过拟合。

除此之外，失活这个思路还可以应用到给网络“瘦身”：通过检测网络中的贡献度比较小的神经元，通过让其永久失活而减小网络的参数量。

#### 12.2你觉得可以对这篇论文有什么改进

与其他防止过拟合的方法进行对比。

以及证明Dropout可以拟模拟多个不同的神经网络的结合。

但是从刚刚的对比图可以发现会有一部分神经元的权重有点相似，也就是说这两个神经元在学习相似的特征。Dropout虽然可以减小网络中的共适性，但是貌似也会增加学习相同特征的神经元（这个现象的理由可能是多几个学习相同特征的神经元，就可以最大程度避免重要的神经元被失活；或者由于真实需要的特征只有那么有限个，但是网络过大参数过多，无法避免学到了相同的特征）

### 13 重要的相关论文

1. D. Cire¸san, U. Meier, and J. Schmidhuber. Multi-column deep neural networks for image classification. Arxiv preprint arXiv:1202.2745, 2012.
2. A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classication with deep convolu-
   tional neural networks. In Advances in Neural Information Processing Systems 25, pages
   1106-1114, 2012.
   Dropout首次使用与Alex Net中。
3. N. Srivastava. Improving Neural Networks with Dropout. Master's thesis, University of
   Toronto, January 2013.
   关于Dropout的一个工作。

### 14 不懂之处

共适性仍然不是特别懂，感觉很模糊。

虽然从第11篇论文（Dropout: A Simple Way to Prevent Neural Networks from Overfitting）中以及证明了这样做是可以起到模拟多个不同的神经网络的结合的作用的，但是这些看似不同的$2^N$个网络会有不少权重是共享的，也就是说是很有可能出现大量的相同的网络？在第11篇论文中确确实实训练了很多个模型来进行结合，从实验结果来看当不同的网络越来越多的时候性能就会超过Dropout（当然，训练这么多个网络的代价很大）。所以也从侧面验证了Dropout只能模拟一部分，并不能等效$2^N$个网络。

### 15 对比

对比论文Dropout: A Simple Way to Prevent Neural Networks from Overfitting，论文的实验部分相对没那么全面。下面是新论文多做了的实验：

* 验证了**p**的影响
* 验证了对稀疏（Sparsity）的影响
* 验证了对数据集的影响
* 在使用的数据集方面是一样的
* 公式证明Dropout的正则化
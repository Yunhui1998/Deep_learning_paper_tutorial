## 8 ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION

### 1 文章想要解决的问题

针对的问题：高维参数空间的随机目标的优化问题。在这种情况下，高阶优化方法是不合适的（太复杂）。因此使用梯度优化更有效，同时也需要考虑噪声。

之前提出的一些典型的优化方法：如随机梯度下降（SGD），dropout正则化。
基于已有算法，提出一种更好的优化算法adam。此算法局限于一阶优化方法。

### 2 研究的是否是一个新问题

不是，但深度学习常常需要大量的时间和机算资源进行训练，这也是困扰深度学习算法开发的重大原因。虽然我们可以采用分布式并行训练加速模型的学习，但所需的计算资源并没有丝毫减少。而唯有**需要资源更少、令模型收敛更快的最优化算法**，才能从根本上加速机器的学习速度和效果，因此提出了新的优化算法——adam算法。

### 3 本文采用的新知识

#### 3.1 ADAM算法：

   ![img](https://upload-images.jianshu.io/upload_images/623192-e8bdddda4ea0a847.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/550)

#### 3.2 有界性
Adam 算法更新规则会很谨慎地选择步长的大小。假定ε=0，则每次时间步t有效下降步长为
$$
\Delta_{t}=\alpha \cdot \hat{m}_{t} / \sqrt{\widehat{v}_{t}}
$$
当$\left(1-\beta_{1}\right)>\sqrt{1-\beta_{2}}$，有$\left|\Delta_{t}\right| \leq \alpha \cdot\left(1-\beta_{1}\right) / \sqrt{1-\beta_{2}}$，反之$\left|\Delta_{t}\right| \leq \alpha$。每一个时间步的有效步长在参数空间中的量级近似受限于步长因子α，即
$$
\left|\Delta_{t}\right| \lesssim \alpha
$$
在当前参数值下确定一个置信域，因此其要优于没有提供足够信息的当前梯度估计。这正可以令其相对简单地提前知道α正确的范围。
提出信噪比（signal-to-noise ratio/SNR）=$\widehat{m}_{t} / \sqrt{\widehat{v}_{t}}$，其大小决定了符合真实梯度方向的不确定性
SNR 值在最优解附近趋向于 0，因此也会在参数空间有更小的有效步长，实现自动退火。

#### 3.3 梯度对角缩放的不变性
有效步长∆t 对于梯度缩放来说仍然是不变量。
$$
\left(c \cdot\hat{m}_{t}\right) /\left(\sqrt{c^{2} \cdot \widehat{v}_{t}}\right)=\widehat{m}_{t} / \sqrt{\widehat{v}_{t}}
$$

#### 3.4 初始偏差修正
解释了指数加权平均数的偏差修正公式的由来。
等号右边为指数加权平均数的完全展开式，为了让指数加权平均数$v_t$和真实值${g_t}^2$更接近，求期望看二者差别。

![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/QY9Ieg3xj8CxXsaRanOTVuR1OFnXe1Q3uQyrxoLLhb2iaFD9cSFrDhKNm7zQfiaHfR2UMedhQKtRb32HCT1YPBMA/640?wx_fmt=png)

#### 3.5 收敛性
利用 Zinkevich 2003 年提出的在线学习框架分析了 Adam 算法的收敛性。
利用附录公式推导出迭代次数无穷时，损失函数必定收敛。

#### 3.6 ADAMAX
将基于$L^2$范数的更新规则泛化到基于$ L^p $范数的更新规则中。虽然这样的变体会因为 p 的值较大而在数值上变得不稳定，但是在特例中，令 p → ∞会得出一个极其稳定和简单的算法ADAMAX。

![img](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/QY9Ieg3xj8CxXsaRanOTVuR1OFnXe1Q3mjKKic7YzySW5EficZt9rgKiakyvJfPGLkJB5ZzyIuzHyp9MvJETK1iblA/640?wx_fmt=png)

和adam相比唯一变动的地方在$u_t$，（注意$m_t$有偏差修正，而$u_t$没有）
其中，不需要修正$ \beta_2$的初始化偏差。同样 AdaMax 参数更新的量级要比 Adam 更简单，即|∆t| ≤ α。
时间平均：由于最后一次迭代有噪声，可以考虑引入指数加权平均数和初始偏差修正。

### 4 相关的关键人物与工作

Adam 通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。 Adam 算法的提出者描述其为两种随机梯度下降扩展式的优点集合，即相关算法：

适应性梯度算法（AdaGrad）：为每一个参数保留一个学习率以提升在稀疏梯度（即自然语言和计算机视觉问题）上的性能。

均方根传播（RMSProp）：基于权重梯度最近量级的均值为每一个参数适应性地保留学习率。这意味着算法在非稳态和在线问题上有很有优秀的性能。

### 5 文章提出的解决方案的关键

Adam 算法同时获得了 AdaGrad 和 RMSProp 算法的优点。

- Adam 不仅如 RMSProp 算法那样基于一阶矩均值计算适应性参数学习率，它同时还充分利用了梯度的二阶矩均值（即有偏方差/uncentered variance）。
- 算法计算了梯度的指数移动均值，超参数$\beta_1$ 和$\beta_2$ 控制了这些移动均值的衰减率。
- 移动均值的初始值和$\beta_1$、$\beta_2$值接近于 1（推荐值），因此矩估计的偏差接近于 0。该偏差通过首先计算带偏差的估计而后计算偏差修正后的估计而得到提升。

### 6 实验设计
评估方法：不同深度学习模型，使用大规模数据集和网络模型，相同的参数初始化，显示结果为最好的超参数。

#### 6.1 Logistic 回归
1.minist数据集，网络为28*28=784，minibatch=128。有学习率衰减
2.IMDB电影评论数据集，测试稀疏特征问题。引入50%dropout噪声防止过拟合。

![image-20201031193559089.png](https://upload-images.jianshu.io/upload_images/24435792-8dca5cbcc36b5ef7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 6.2  多层全连接神经网络测试非凸目标函数。
MINIST训练集，两层完全连接的隐含层,每层隐含单元1000,ReLU激活的神经网络模型, minibatch size=128。引入L2正则化。
另外，将adam 和在多层神经网络的优化上显示了良好的性能的 SFO拟牛顿法进行对比，两者都引入dropout正则化。

![image-20201031193534403.png](https://upload-images.jianshu.io/upload_images/24435792-5154f27389421111.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 6.3 卷积神经网络

CNN：5x5卷积核和3x3 max pooling,步长为2,然后是个由1000个线性隐藏单元组成的全连接层(ReLU),对输入图像进行了白化预处理，在输入层和全连接层应用了dropout噪声。minibatch的大小设置为128，结果对比如图所示。

![image.png](https://upload-images.jianshu.io/upload_images/24435792-d05d33150723e3e4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

图中，Adagrad前期收敛速度比SGD快速，而最终SGD算法和ADAM算法比Adagrad算法效果要好。实际上在cnn中，一次方矩阵对加速起重要影响，而二次矩阵效果一般。这对挑选优化算法提出了指导意见。

#### 6.4 偏差校正

探讨偏差修正对算法的影响。由于引入动量的RMSProp没有偏差校正，和Adam算法对比。

利用一种变分自编码器 VAE进行实验，调整不同的衰减率和学习率参数下，性能测试。

![image.png](https://upload-images.jianshu.io/upload_images/24435792-ec1bc737c75b6ba3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 7 采用的数据集及其是否开源

开源代码：https://github.com/michaelshiyu/kerNET

### 8 实验结果是否验证了科学假设

是，具体可参照第六节，依据实验证明了ADAM算法的优点，如第九节所示。

### 9. 本文贡献

Adam 是一种可以替代传统随机梯度下降过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重。Adam 最开始是由 OpenAI 的 Diederik Kingma 和多伦多大学的 Jimmy Ba 在提交到 2015 年 ICLR 论文（Adam: A Method for Stochastic Optimization）中提出的。

首先该算法「Adam」的名称来源于适应性矩估计（adaptive moment estimation）。在介绍这个算法时，原论文列举了将 Adam 优化算法应用在非凸优化问题中所获得的优势：
- 实现简单，高效的计算，所需内存少
- 梯度对角缩放的不变性（2中将给予证明）
- 适合解决含大规模数据和参数的优化问题
- 适用于非平稳（non-stationary）目标/非凸优化（实验验证）
- 适用于解决包含很高噪声或稀疏梯度的问题（实验验证）
- 超参数可以很直观地解释，并且基本上只需极少量的调参

### 10 下一步怎么做（本文局限性）

可以探究adam超参数设置，调参的经验
缺点改进：adam虽然收敛的很快，也很稳定，但是收敛的效果差（即收敛到的最优解的准确率偏低）。
提出更好的优化算法，比如AMSGrad和AdaBound。后者是目前来说比较有希望顶替Adam的：前期和Adam一样快，后期有和SGD一样的精度。

### 11 重要的相关论文

与Adam算法有直接联系的优化方法

- RMSProp(Tieleman&Hinton,2012;Graves,2013)

  Hinton教授讲述RMSProp算法的材料：http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

- AdaGrad(Duchietal.,2011)：https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

### 12 不懂之处

1. 对论文证明ADAM算法的收敛性，大段数学公式难以理解
2. 如何选择ADAM优化器？可以参照这篇博客：https://blog.csdn.net/qq_35860352/article/details/80772142


### 13 专业术语

1. 指数加权平均（bias correction in exponentially weighted average）

![在这里插入图片描述](https://i2.wp.com/img-blog.csdnimg.cn/20200430115301310.png)

2. 动量（momentum）

![在这里插入图片描述](https://i2.wp.com/img-blog.csdnimg.cn/2020043012073163.png)

3. 梯度对角缩放的不变性（Invariance of gradient diagonal scaling）

​        有效步长∆t 对于梯度缩放来说仍然是不变量。

![img](https://upload-images.jianshu.io/upload_images/623192-27a4dfbcb320aae0.png?imageMogr2/auto-orient/strip|imageView2/2/w/321/format/webp)

4. 稀疏特征/稀疏梯度（sparse features/gradients）

   具有稀疏特征平稳段会减缓学习，平稳段是一块区域，其中导数长时间接近于0，梯度会从曲面从从上向下下降。因为稀疏特征的0很多，导致平均梯度会很小，到达鞍点，再继续下降，进而导致网络对于稀疏特征的学习很慢（对比于更密集的特征）。

![4.png](https://upload-images.jianshu.io/upload_images/24435917-8ebef95ccc44367d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 14 英语词汇

| 英文                           | 中文/解释                                               |
| ------------------------------ | ------------------------------------------------------- |
| adaptive moment estimation     | 自适应矩估计：基于不同阶数矩阵来适应不同参数的学习速率  |
| sparse features/gradients      | 稀疏特征/稀疏梯度                                       |
| exponential moving average     | 指数移动平均数                                          |
| initialization bias correction | 初始偏差修正                                            |
| over-fitting                   | 过拟合                                                  |
| robust,robustness              | 稳健性/鲁棒性：一个系统或组织有抵御或克服不利条件的能力 |
| normalization                  | 规范化/归一化/标准化                                           |


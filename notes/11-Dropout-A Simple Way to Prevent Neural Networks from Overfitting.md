# Dropout: A Simple Way to Prevent Neural Networks from Overfitting

### 1 文章想要解决的问题

#### 1.1 过拟合（Overfitting）

​	过拟合具体表现在：模型在训练数据上损失函数较小，预测准确率较高；但是在测试数据上损失函数比较大，预测准确率较低。

![img](https://img-blog.csdn.net/20181012150407211?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NvbmdjaHVueGlhbzE5OTE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

​	常用的解决过拟合方法是**bagging**，其具体局限如下：boosting

* 训练一个大型神经网络需要很大的数据集，数据集不够大将导致无法划分足够大的数据子集。
* 由于超参数的存在，训练一个大型神经网络通常非常**耗时**。
* 同时运行多个神经网络通常运行效率低下，并且会**消耗许多计算资源**。

### 2 研究的是否是一个新问题

​		不是，研究的是如何解决神经网络过拟合问题。

​		在2012年，Hinton在其论文《Improving neural networks by preventing co-adaptation of feature detectors》中提出Dropout。当一个复杂的前馈神经网络被训练在小的数据集时，容易造成过拟合。为了防止过拟合，可以通过阻止特征检测器的共同作用来提高神经网络的性能。

​		在2012年，Alex、Hinton在其论文《ImageNet Classification with Deep Convolutional Neural Networks》中用到了Dropout算法，用于防止过拟合。并且，这篇论文提到的AlexNet网络模型引爆了神经网络应用热潮，并赢得了2012年图像识别大赛冠军，使得CNN成为图像分类上的核心算法模型。

​		在本文中，对**Dropout**进行一个详细的介绍，即一种通过在神经网络的隐藏单元中加入噪声使其正则化的方法。文章和之前相关正则化工作相比，创新点主要有：

#### 2.1  扩展应用领域

​		Vincent et al.(2008, 2010)曾在去噪自动编码器(DAEs)的环境中使用过在单位状态中添加噪声的概念将其加入到自动编码器的输入单元中，训练网络重构无噪声输入。文章的工作扩展了这一思想，表明dropout也可以有效地应用于隐藏层，还证明了添加噪声不仅对无监督特征学习有用，而且可以推广到有监督学习问题。

#### 2.2  探索隐藏单元

​		van der Maaten等人(2013)研究了不同指数族噪声分布对应的正则化器，包括dropout(drop - out noise)(称之为“blankout noise”)。但是，它们对输入应用噪声，并且只探索没有隐藏层的模型。Wang和Manning(2013)提出了一种通过消除Dropout噪声来加速Dropout的方法。Chen等(2012)在去噪自编码器的背景下探讨了消除问题。

​		在dropout中，随机地最小化噪声分布下的损失函数。这可以看作是最小化预期损失函数。Globerson和Roweis先前的工作(2006)、Dekel等人(2010)研究了另一种情况，即当adversary选择放弃哪些单位时，损失最小。其中可以丢弃的单元的最大数量不是噪声分布，而是是固定的。然而，这项工作也没有探索隐藏单元的模型。

### 3 本文采用的知识

#### 3.1 正则化

​		正则化技术是保证算法泛化能力的有效工具,它可以令参数数量多于输入数据量的网络避免产生过拟合现象。

##### 3.1.1 数据增强

​		数据增强是提升算法性能、满足深度学习模型对大量数据的需求的重要工具。数据增强通过向训练数据添加转换或扰动来人工增加训练数据集。数据增强技术如水平或垂直翻转图像、裁剪、色彩变换、扩展和旋转通常应用在视觉表象和图像分类中。

##### 3.1.2 L1和L2正则化

​		正则化(regularization)的思想是在损失函数中加入刻画模型复杂程度的指标。L1 和 L2 正则化是最常用的正则化方法。L1 正则化向目标函数添加正则化项，以减少参数的绝对值总和；而 L2 正则化中，添加正则化项的目的在于减少参数平方的总和。根据之前的研究，L1 正则化中的很多参数向量是稀疏向量，因为很多模型导致参数趋近于 0，因此它常用于特征选择设置中。机器学习中最常用的正则化方法是对权重施加 L2 范数约束。

##### 3.1.3 早停法(Early Stopping)

​		在交叉验证集的误差上升之前的点停止迭代，避免过拟合。如下图所示，最优模型是在垂直虚线的时间点保存下来的模型，即处理测试集时准确率最高的模型。

![这里写图片描述](https://img-blog.csdn.net/20180104230552110?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMjA5MDkzNzc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 3.2 MNIST实验中采用了max-norm constraint（最大范式约束）

​		max-norm constraint，即最大范式约束，最大范式约束指的是给每个神经元的权重向量的量级设置上限，用公式来表示为：
​																		<img src="https://ae04.alicdn.com/kf/Hb5fc2978905b431aacdc9374ddcc3ab9A.png" alt="image.png" title="image.png" />
​		一般c值为3或者4,该算法有个优点，无论学习率多高，都不会出现爆炸现象，因为权重w始终小于c。

#### 3.3 蒙特卡洛模型（Monte-Carlo Model）

​		蒙特卡罗模型是一种随机模拟方法。以概率和统计理论方法为基础的一种计算方法。将所求解的问题同一定的概率模型相联系，用电子计算机实现统计模拟或抽样，以获得问题的近似解。为象征性地表明这一方法的概率统计特征，故借用赌城蒙特卡罗命名。原理是通过大量随机样本，去了解一个系统，进而得到所要计算的值。

### 4 相关的关键人物与工作

#### 4.1 相关的关键人物

##### 4.1.1 Nitish Srivastava

​		该论文的一作，另一个重要贡献是发表该论文之前就提出了关于减小网络中共适性（co-adaptation）的一项工作——Dropout。Nitish Srivastava曾与Hinton一起做研究，在毕业后去了苹果工作。

![th (197×224) (bing.net)](https://tse1-mm.cn.bing.net/th?id=OIP.rnNuSjSz4_3_T1MYvr8p1gAAAA&w=141&h=160&c=8&rs=1&qlt=90&dpr=1.4&pid=3.1&rm=2)

以下是他影响力比较大的论文。

<img src="https://ae02.alicdn.com/kf/H0b302a195ee244dba4d02db00bdada25N.png" alt="image.png" title="image.png" />

##### 4.1.2 Geoffrey Everest Hinton

​		加拿大认知心理学家和计算机科学家，被誉为“神经网络之父”、“深度学习鼻祖”、”人工智能教父“。现任Google副总裁兼工程研究员、多伦多大学的特聘教授，也是Vector Institute首席科学顾问。2018年因作为“深度学习领域的三大先驱之一”获得图灵奖，被选为2017年改变全球商业格局的50人。在Alex Net的论文里介绍过了。

<img src="https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1605774694262&di=bb2f1207f539a8cb6858663af5d1a641&imgtype=0&src=http%3A%2F%2Fpic2.zhimg.com%2F50%2Fv2-1d49524120dcde04c5f00068cb78b810_hd.jpg" style="zoom:80%;" />

##### 4.1.3 Alex Krizhevsky

​		引用了两次，同时也是作者之一。

<img src="https://ae05.alicdn.com/kf/Hc6965346d96742ed89f668b1befeffcbJ.png" alt="image.png" title="image.png" />

​		上次在Alex Net的论文里介绍过了。

##### 4.1.4 Yann LeCun

​		文中被引用了三次，主要引用了他关于识别和分类的论文。同样在Alex Net的论文里介绍过了。

##### 4.1.5 Yoshua Bengio

![yoshua-benjio-1.jpg (898×600) (msra.cn)](https://www.msra.cn/wp-content/uploads/2018/11/yoshua-benjio-1.jpg)

​		三巨头之一，最具代表性的工作之一是LeNet，将反向传播用于卷积神经网络；其次是开创了神经网络做语言模型的先河，比如word2vec等；以及近两年在Attention和GAN上做的工作，都非常重要。

<img src="https://ae03.alicdn.com/kf/H1b08dedbd1924c20937ba5a1a3422bc7m.png" alt="image.png" title="image.png" />

​		本文中引用了3次，主要引用了他关于去噪工作的论文。

#### 4.2 相关的关键工作

##### **4.2.1 AlexNet**

​		Dropout第一次应用是在Alex Net中为了防止过拟合而添加的。但是关于Dropout的原理、为什么有用等细节问题是后来才慢慢重视起来的，并且也后续发表了不少相关的论文。

### 5 文章提出的解决方案的关键

#### 5.1 **Dropout类似于性别在生物进化中的角色**

​		Dropout类似于性别在生物进化中的角色，物种为了生存往往会倾向于适应这种环境，环境突变则会导致物种难以做出及时反应，性别的出现可以繁衍出适应新环境的变种，有效的阻止过拟合，即避免环境改变时物种可能面临的灭绝。

#### 5.2 **减少神经元之间复杂的共适应关系**

​		因为dropout导致两个神经元不一定每次都在一个dropout网络中出现，这样权值的更新不再依赖于有固定关系的隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况 。迫使网络去学习更加鲁棒的特征 ，这些特征在其它的神经元的随机子集中也存在。换句话说假如我们的神经网络是在做出某种预测，它不应该对一些特定的线索片段太过敏感，即使丢失特定的线索，它也应该可以从众多其它线索中学习一些共同的特征。

#### 5.3 **取平均的作用**

​		dropout掉不同的隐藏神经元就类似在训练不同的网络，随机删掉一半隐藏神经元导致网络结构已经不同，整个dropout过程就相当于对很多个不同的神经网络取平均。而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互抵消就可以达到整体上减少过拟合。

### 6 实验设计

#### 6.1 模型设计

​		考虑具有L个隐藏层的神经网络。设l∈{1，…，L}为网络的隐层提供索引。设z（l）表示输入到第l层的向量，y（l）表示第1层（y（0）= x是输入）的输出向量。 W（1）和b（1）是第1层的权重和偏差。 

​		公式中的f是激活函数，例如，f是sigmoid函数， f(x) = 1/(1 + exp(−x)).

​				标准神经网络

<img src="https://ae03.alicdn.com/kf/Hea4517de0c1d4a018254a9c26b4e9931k.png" alt="image.png" title="image.png" />

​				加上Dropout神经网络

<img src="https://ae03.alicdn.com/kf/H70aada8fd6ba488b9c95f55bf0ede293x.png" alt="image.png" title="image.png" />

​		上面公式中Bernoulli函数是为了生成概率r向量，也就是随机生成一个0、1的向量。		

​		代码层面实现让某个神经元以概率p停止工作，其实就是让它的激活函数值以概率p变为0。比如我们某一层网络神经元的个数为1000个，其激活函数输出值为y1、y2、y3、......、y1000，我们dropout比率选择0.4，那么这一层神经元经过dropout后，1000个神经元中会有大约400个的值被置为0。





<img src="https://ae03.alicdn.com/kf/H6fe20968128d46b4b052a7247b599a89l.png" alt="image.png" title="image.png" />

​		需要注意的是采用Dropout，训练的时候只有占比为 ![[公式]](https://www.zhihu.com/equation?tex=p) 的隐藏层单元参与训练，那么在预测的时候，如果所有的隐藏层单元都需要参与进来，则得到的结果相比训练时平均要大 <img src="https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bp%7D" alt="[公式]" style="zoom:80%;" />，为了避免这种情况，就需要测试的时候将输出结果乘以 ![[公式]](https://www.zhihu.com/equation?tex=p) 使下一层的输入规模保持不变。这个被称为**inverted dropout**。

<img src="https://ae05.alicdn.com/kf/H2ae466009b06475ab5254bee0f22522aJ.png" alt="image.png" title="image.png" />

#### 6.2 分析结果

##### 6.2.1 MNIST

​		MNIST数据集由28×28像素的手写数字图像组成。任务是将图像分类成10位数的类别。实验结果如下：

<img src="https://ae04.alicdn.com/kf/Hc0138c7b2ded44fb8c8518c40d8de98aZ.png" alt="image.png" title="image.png" />

​		为了检验dropout的鲁棒性，保持所有的超参数（包括p）固定，用许多不同的架构网络进行分类实验。

<img src="https://ae02.alicdn.com/kf/Hf364623ba8984b4b99fc6f356ef367e9R.png" alt="image.png" title="image.png" />

##### 6.2.2 Street View House Numbers

​		街景房屋号码（SVHN）数据集由谷歌街景收集的房屋号码的彩色图像组成，例子如下所示：

<img src="https://ae03.alicdn.com/kf/H2ca95cd25b5144f3b043124b4d59091e9.png" alt="image.png" title="image.png" />

实验结果如下：

<img src="https://ae05.alicdn.com/kf/H73ba68ec2691477ca6adba042aa4acafT.png" alt="image.png" title="image.png" />

##### 6.2.3 CIFAR-10 and CIFAR-100

​		CIFAR-10和CIFAR-100数据集由分别来自10个和100个类别的32×32个彩色图像组成。例子如下所示：

<img src="https://ae03.alicdn.com/kf/H033a25cf584746d5acb5320b708003a2x.png" alt="image.png" title="image.png" />

实验结果如下：

<img src="https://ae03.alicdn.com/kf/H1dae6286275a4865a4ed010d307eee1f5.png" alt="image.png" title="image.png" />

##### 6.2.4 ImageNet

​		ImageNet是一个超过1500万标记的高分辨率图像数据集，属于大约22000个类别。从2010年开始，作为Pascal视觉对象挑战赛的一部分，每年举办一次名为ImageNet大型视觉识别挑战赛（ILSVRC）的比赛。在这个挑战中使用了ImageNet的一个子集，1000个类别中大概有1000个图像。本文中主要用的是ILSVRC-2010：http://www.image-net.org/challenges/LSVRC/2010/。

<img src="https://ae04.alicdn.com/kf/H844475806897449683033005dc85f993K.png" alt="image.png" title="image.png" />

实验结果如下：

<img src="https://ae01.alicdn.com/kf/Hbb716120d99649bb96d892b178cf1a54Y.png" alt="image.png" title="image.png" />

##### 6.2.5 TIMIT

​		将Dropout应用于语音识别任务，TIMIT数据集由680位发言者的录音组成，涵盖了美国英语的8种主要方言，在受控制的无噪音环境下阅读10个语音丰富的句子。实验结果如下：

<img src="https://ae02.alicdn.com/kf/H1510e3c93b19440f816234953e84eebe1.png" alt="image.png" title="image.png" />

##### 6.2.6 Results on a Text Data Set

​		将Dropout应用于文字领域，Reuters-RCV1数据集的一个子集，收集了来自路透社的超过800,000篇newswire文章。

​		实验结果：没有使用dropout的神经网络获得了31.05％的错误率。 添加dropout将错误降低到29.62％。 与视觉和语音数据集相比，这一改进要小得多。

##### 6.2.7 Comparison with Bayesian Neural Networks

​		与贝叶斯网络做对比实验，数据集使用的是Alternative Splicing。其任务是根据RNA特征来预测选择性剪接的发生。评估指标是代码质量，它是目标与预测概率分布之间的负KL散度的量度（越高越好）。实验结果如下：

<img src="https://ae01.alicdn.com/kf/Hd47052a4099c479db706525b496180d8z.png" alt="image.png" title="image.png" />

##### 6.2.8 Comparison with Standard Regularizers

​		与标准正则化比较，数据集为MNIST，实验结果如下：

<img src="https://ae04.alicdn.com/kf/H17e350e9627a43fe8b0841f1ba32b4baR.png" alt="image.png" title="image.png" />



#### 6.3 Salient Features

​		在MNIST数据集上，对比了Dropout和其他正则化方法，如下表可以看出，Dropout在测试分类错误率更低。

<img src="https://ae04.alicdn.com/kf/H18618ff4446242b5a2b3dd0c6fc64c06J.png" alt="image.png" title="image.png" />

##### 6.3.1 对特征图谱的影响

​		采用Dropout，隐藏的单元可以检测到其不同部分的边缘，笔触和斑点。 这表明Dropout确实会破坏共同适应，这可能是主要的导致较低的泛化错误的原因。（图没看懂）

<img src="https://ae03.alicdn.com/kf/H43c5fc0ad26e4bd48d6eefb612b9dbcfN.png" alt="image.png" title="image.png" />

##### 6.3.2 稀疏性的影响

​		可以看出，使用Dropout后 很多隐层单元的activation 激活值都接近与0，只有少部分的神经元的激活值特别大。

<img src="https://ae05.alicdn.com/kf/H681906a51e1746868749f1473cae8a75V.png" alt="image.png" title="image.png" />

##### 6.3.3 p的影响

​		左图表示 隐层节点不变的情况下，当p增加时，测试误差先下降，然后再在[0.4,0.8]中平滑，最后在0.8~1时 误差又上升，使用底部很宽的”U”形曲线，所以一般让隐层的p取0.5就是最优的。

<img src="https://ae02.alicdn.com/kf/H122f6de747cf49e8a162cb66a5e2341af.png" alt="image.png" title="image.png" />

##### 6.3.4 数据集大小的影响

​		可以看到，在数据集较小时，不采用dropout的标准神经网络表现较好，当数据集再继续增大时以上时，使用dropout的优势就开始凸显了。

<img src="https://ae03.alicdn.com/kf/H4dcc3c32570844c1b2259362b3d81329P.png" alt="image.png" title="image.png" />

##### 6.3.5 Monte-Carlo Model Averaging vs. Weight Scaling

​		Dropout在训练和测试时使用的算法很明显不一致，前者用的是随机失活，而后者则是简单地用$p$放缩权重。相对地，如果测试时采用与训练时相似的算法，那应该测试一个样本时也采用随机失活，并且是采用多次然后取平均值，这样才能比较真实地反应学习的效果。但是很明显，这样做效率太低了，这也是为什么论文最后采用了放缩权重的方式的缘故。

​		所以，论文在此处做了个实验，证明测试时直接放缩权重这个更简便的操作能做到近似平均的效果。下图横轴是采用的模型数量$k$，纵轴是测试集的错误率；蓝线是对$k$个模型采用蒙特卡洛平均的结果，红线是直接放缩权重的结果；中心点处伸出来的上下界是指那次实验数据的极大极小值。

​		可以发现在$k$约为50的时候两者是最接近的。虽然$k$接近无穷时应该是会比放缩权重更准确地描述错误率（实际上$k$在大于60时就已经更准确了），但是计算代价太大，所以采用简单的权重缩放也是可行的。

<img src="https://ae03.alicdn.com/kf/H014c70d1c2f74d4e8b85e053f5c43170y.png" alt="image.png" title="image.png" />

### 7 采用的数据集及其是否开源

是，具体如下：

* Street View House Numbers data set(SVHN)：http://ufldl.stanford.edu/housenumbers/

* ImageNet：http://www.image-net.org/   （本文中主要用的是ILSVRC-2010：http://www.image-net.org/challenges/LSVRC/2010/）

* CIFAR-100：https://www.cs.toronto.edu/~kriz/cifar.html

* MNIST：http://yann.lecun.com/exdb/mnist/

* TIMIT：https://catalog.ldc.upenn.edu/LDC93S1

* Reuters-RCV1：http://archive.ics.uci.edu/ml/datasets/reuters+rcv1+rcv2+multilingual,+multiview+text+categorization+test+collection

* Alternative Splicing data set：http://prosplicer.mbc.nctu.edu.tw/

  <img src="https://ae04.alicdn.com/kf/Hef25875a741b4c7e9ba4e430e2c6ba5fK.png" alt="image.png" title="image.png" />

### 8 实验结果是否验证了科学假设

​		是。Dropout在SVHN，ImageNet，CIFAR-100和MNIST上获得最新的结果，验证了Dropout的通用性和有效性。

### 9 本文贡献

1. 详细解释了Dropout；
2. 对Dropout进行了泛化的说明；
3. 做了许多的实验来验证Dropout的有效性。

### 10 下一步怎么做

* 使用Dropout的缺点就是会使得训练时间增加，是没使用Dropout网络的2-3倍，因此，加速Dropout也是未来一个研究点。一个在减少时间的同时也获得dropout效果的方法就是加入一个和Dropout效果相同的惩罚项。（类似yolo v3，应用于工程领域）

* 注意力机制，核心目标是从众多信息中选择出对当前任务目标更关键的信息。

  注意力机制（Attention Mechanism）源于对人类视觉的研究。在认知科学中，由于信息处理的瓶颈，人类会选择性地关注所有信息的一部分，同时忽略其他可见的信息。上述机制通常被称为注意力机制。人类视网膜不同的部位具有不同程度的信息处理能力，即敏锐度（Acuity），只有视网膜中央凹部位具有最强的敏锐度。为了合理利用有限的视觉信息处理资源，人类需要选择视觉区域中的特定部分，然后集中关注它。例如，人们在阅读时，通常只有少量要被读取的词会被关注和处理。综上，注意力机制主要有两个方面

  - 决定需要关注输入的哪部分。

  - 分配有限的信息处理资源给重要的部分。

    （blog：http://kakuguo.ink/2020-03-18-Summary-of-Computer-Vision-Attention/）

  （综述论文）Hu D. An Introductory Survey on Attention Mechanisms in NLP Problems[J]. arXiv preprint arXiv:1811.05544, 2018.

* 可解释性，为什么丢弃这些就能起作用，采取的策略等等。

* 借鉴其他学科思路，交叉寻求灵感。

* 压缩模型

  * Ghostnet-shuffleNetV3

### 11 重要的相关论文

1. D. Cire¸san, U. Meier, and J. Schmidhuber. Multi-column deep neural networks for image classification. Arxiv preprint arXiv:1202.2745, 2012.

   这个模型主要的内容就是一个特殊的bagging＋cnn，因此叫做multi-column，每一个column都是一个cnn。

   对于图像的处理，每一个cnn都是对于一种preprocessor之后的图片进行处理，最后的softmax分布结构求和之后求平均。bagging是投票求和，softmax分布也可以看做是投票。

   

2. A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classication with deep convolu-
   tional neural networks. In Advances in Neural Information Processing Systems 25, pages
   1106-1114, 2012.
   Dropout首次使用与Alex Net中。

### 12 单词

prohibitively 静止地；过高地；过分地

exponentially 以指数方式

robust 健壮的，鲁棒

conspiracy 阴谋

max-norm constraint 最大范式约束

Salient 突出

co-adaptation  共适性

marginalization 边缘化

inverted 倒

deterministic counterpart	原图

adversary 对手

Monte-Carlo 蒙特卡罗

#### 13 参考博客

* https://zhuanlan.zhihu.com/p/38200980

* https://blog.csdn.net/qq_25011449/article/details/81168369
* https://blog.csdn.net/weixin_37993251/article/details/88176217
* https://www.sohu.com/a/327933073_468740 
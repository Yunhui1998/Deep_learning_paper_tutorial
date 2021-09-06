## 3 ImageNet Classifification with Deep Convolutional Neural Networks

这篇论文讲的是大神Alex Krizhevsky、Ilya Sutskever、Geoffrey E. Hinton三人提出的**AlexNet**深度卷积神经网络，摘得了2012年ILSVRC比赛的桂冠，该文章的重要意义在于其在ImageNet比赛中以巨大的优势击败了其它非神经网络的算法，在此之前，神经网络一直处于不被认可的状态。

### 1 文章想要解决的问题

#### 1.1 训练具有强大学习能力的深度卷积神经网络模型

现实环境中的物体可能表现出相当大的变化性，要学会识别这些物体就必须使用更大的训练集，也就更需要一个更大更复杂的、学习能力更强的模型。

故本文训练了一个庞大的深度卷积神经网络AlexNet，将ImageNet LSVRC-2010比赛中的120万张高分辨率图像分为1000个不同的类别并且在测试数据上取得了37.5％和17.0％的前1和前5的错误率，比以前的顶尖水平还要好得多。

- 网络输入：将ImageNet数据集的图像下采样为256×256的固定分辨率图像输入。
- 网络输出：softmax输出网络对该输入图像分别属于1000个类别的预测概率。

#### 1.2 提高训练速度

1. 采用ReLU非线性单元替代原来的tanh激活函数，训练速度提高好几倍；
2. 采用两个CPU并行化操作而不是单个CPU，降低了错误率且减少了训练时间。

#### 1.3 面对更大规模的数据集，如何有效避免过拟合

方法1：数据增强(平移图像和水平映射、改变训练图像中RGB通道的灰度)；

方法2：Dropout正则化。本文选择在前两个全连接层使用**dropout**以避免过拟合。

### 2 研究的是否是一个新问题

是，本网络结构是开启ImageNet数据集更大、更深CNN的开山之作。本文对CNN的一些改进成为以后CNN网络通用的结构，之后在ImageNet上取得更好结果的ZF-net、SPP-net、VGG等网络，都是在其基础上修改得到的。

本网络能获得成功的因素是：

1. 使用ReLU代替tanh作为激活函数，成功解决了tanh在网络较深时的梯度弥散问题；
2. 利用GPU并行化操作加速深度卷积网络的训练；
3. 提出了LRN层。LRN全称Local Response Normalization，对局部神经元创建竞争机制，增大响应大的单元，抑制反馈小的神经元，增强了模型的泛化能力。
4. 使用了重叠最大池化操作，即池化步长比池化核尺寸小，提升了特征的丰富性。此前CNN普遍采用平均池化，最大池化能够避免平均池化的模糊化效果；
5. 最后几个全连接层使用Dropout，避免过拟合；
6. 数据增强。通过随机裁剪、旋转、翻转等操作，减轻过拟合，提升模型泛化性能。

### 3 本文采用的新知识

#### 3.1 ReLU非线性单元

AlexNet使用的神经元激活函数是ReLU激活函数 $f(x)=max(0,x)$ ，相比于饱和非线性函数如Sigmoid和tanh函数，不饱和非线性函数如ReLU在梯度下降时具有**更快的收敛速度**，更快地学习，在大型网络中训练大量数据具有非常好的效果。作者通过在CIFAR-10数据集上做的实验证明了此论点。如下图，实线表示使用ReLUs的CNN，虚线表示使用tanh函数的CNN，可以看出使用ReLUs的CNN能更快地把训练错误率降低到25%（迭代次数比tanh的CNN快约5倍）。

![](https://img-blog.csdnimg.cn/20190527110048194.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA1NjU5MTA=,size_16,color_FFFFFF,t_70)

ReLU是本文作者Hinton在2010年提出来改善RBM性能的，把ReLU引入深度CNN中，使得ReLU成为以后深度网络普遍使用的非线性函数，从而在这个领域替代经典的sigmoid、tanh函数。ReLU有三个好处：

1. 简单的max计算，大大减少了计算量，可以提高训练速度；
2. 梯度在ReLU中是直接传递的，鉴于深度网络的梯度衰减现象，ReLU可以保持梯度，减缓梯度衰减的趋势；
3. bp过程中没有了梯度换算的操作，加快了训练。

#### 3.2 跨GPU并行化操作

一个GTX580的内存只有3GB，有限的内存限制了可以在GPU上训练的最大网络。目前的GPU很适合于跨GPU并行化操作，故作者把网络一分为二，分配到2个GPU上，通过并行计算来解决，不用通过主机的缓存，当前GPU相互间可以很好进行读写操作。

这里作者有一个小技巧，GPU相互“沟通”：例如，网络中layer-3的filter把layer-2的所有特征图作为输入，而其它卷积层，只从同一个GPU内的上一层特征图作为输入。为什么layer-3把layer-2的全部特征图作为输入而其它层却不这样，这个作者并没有解释理论依据，通过交叉验证实验得来的。最终的结构有点类似Cresian提出的多列卷积网络，但是本文的网络不是完全独立的，这种方式可以提高1.2%的准确率。

#### 3.3 局部响应归一化

LRN（Local Response Normalization）层是用来做归一化的。ReLU函数不需要归一化来防止饱和现象，即对于很大的输入x，ReLU仍然可以有效的学习，但是作者发现即使这样，对数据进行局部归一化对于学习来说还是有帮助的，可以增强模型的泛化能力。

局部响应归一化公式：

<img src="https://img-blog.csdn.net/20150127210049765" style="zoom:50%;" />

具体方式：

选取临近的n个特征图，在特征图的同一个空间位置（x,y）依次平方，然后求和，在乘以 $\alpha$，在加上 $k$。这个局部归一化方式与论文“What is the best multi-stage architecture for Object Recognition”中的局部归一化方法不同：本文的归一化只是多个特征图同一个位置上的归一化，属于特征图之间的局部归一化（属于纵向归一化），作者命名为亮度归一化；“what……”论文中在特征图之间基础上还有同一特征图内邻域位置像素的归一化（横向、纵向归一化结合）；“what……”归一化方法计算复杂，但是没有本文中 $\alpha、k、n$ 等参数，本文通过交叉验证来确定这三个参数；此外，本文的归一化方法没有减去均值，感觉是因为ReLU只对正值部分产生学习，如果减去均值会丢失掉很多信息。

简言之，LRN是“What is the best multi-stage architecture for Object Recognition”论文使用的方法，本文简化改进了该归一化方法，将模型错误率降低1.2%左右。

#### 3.4 重叠池化

正常池化是**步长 = 窗口尺寸**如步长s=2、窗口z=2，重叠池化是指**步长＜窗口尺寸**如步长s=2、窗口z=3。

Pooling层一般用于降维，将一个 $k \times k$ 的区域内取平均或取最大值，作为这一个小区域内的特征，传递到下一层。传统的Pooling层是不重叠的，而本论文提出使Pooling层重叠可以降低错误率，而且对防止过拟合有一定的效果。

个人理解，使Pooling层重叠，可以减少信息的损失，所以错误率会下降。但是对防止过拟合的效果，文章也只是说slightly，目测意义不大。

#### 3.5 Dropout

Dropout是Hinton在2012年提出以改善深度网络泛化能力的。文中采用Dropout技术，在每次训练网络时，每个神经元都会以0.5的概率“丢失”，丢失的神经元不参与前向传播和反向传播，但是神经元的权值在每次训练时都是共享的。这个技术降低了神经元之间的联合适应性，从而学习更具鲁棒性的特征。在测试阶段，神经网络会使用所有的神经元，但每个神经元的输出会乘以0.5。Dropout技术用在前两个全连接层中，会减少过拟合但有个缺点，就是会使最终网络收敛所需要的迭代次数翻倍。

### 4 相关的关键人物与工作

#### 4.1 相关的关键人物

对这篇文章所引用的参考文献的作者进行统计，发现引用文献最多的作者分别是L. Fei-Fei (4次)、A. Krizhevsky (4次)、Y. LeCun (4次)、G.E. Hinton (3次)、J. Deng (3次)，故可将这5人作为与本文研究工作相关的关键人物。

##### 4.1.1 Li Fei-Fei （李飞飞）

现为美国斯坦福大学计算机科学系教授、美国国家工程院院士、以人为本人工智能研究院（HAI）院长、AI4ALL联合创始人及主席 、Twitter公司董事会独立董事，主要研究方向为机器学习、计算机视觉、认知计算神经学，曾获斯隆研究奖计算机科学奖、影响世界华人大奖，入选2015年“全球百大思想者”、“2017年度中国留学人员50人榜单”。2015年在TED的一段演讲“[李飞飞：如何教计算机理解图片](https://www.ted.com/talks/fei_fei_li_how_we_re_teaching_computers_to_understand_pictures?language=zh-cn)”很火爆。

<img src="https://bkimg.cdn.bcebos.com/pic/c75c10385343fbf28136b56dba7eca8065388f61?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2U5Mg==,g_7,xp_5,yp_5" style="zoom:60%;" />

最突出的贡献：构建了**[ImageNet]([http://www.image-net.org/])**数据集

> **李飞飞在数据方面的研究改变了人工智能研究的形态，从这个意义上讲完全称得上是「改变了世界」。**

- **ImageNet:**

ImageNet项目由李飞飞教授于2007年发起，其团队花费两年半的时间才完成了一个含有**1500万张照片、涵盖22000种物品**的数据库，在2009年发表了一篇名为《ImageNet: A Large-Scale Hierarchical Image Database》的论文并免费公开数据集，但当时并没有激起很大的浪花甚至人们对更多的数据能改进算法这种简单的概念还相当怀疑。

ImageNet的重大转折点是**ImageNet挑战赛**：李飞飞说服著名的图像识别大赛 PASCAL VOC 举办方与 ImageNet 联合举办赛事，虽然 PASCAL 大赛虽然备受瞩目，数据集的质量很高，但类别却很少，只有 20 个，相比之下，ImageNet 的图像类别高达 1000个。随着比赛不断举办，ImageNet 与 PASAL 合作的这项赛事成为衡量图像分类算法在当时最复杂图像数据集上性能如何的一个基准。ImageNet挑战赛从2010年开始，到2017年不再举办，而后ImageNet由Kaggle公司继续维护。

> “人们惊讶地发现经 ImageNet 训练后的模型可以用作其它识别任务的启动模型。你可以先用 ImageNet 训练模型，然后再针对其它任务调试模型，这不仅仅是神经网络领域的突破，也是识别领域的重大进展。”
>
> **ImageNet 真正改变了人工智能领域对「数据」的认知，它让人们真正意识到数据集在 AI 研究中的核心地位，它和算法同等重要。**
>
> 参考：https://www.zhihu.com/question/30990652

##### 4.1.2 Jia Deng

李飞飞的博士生，本科是清华大学计算机科学专业，现任密歇根大学计算机科学与工程系的助理教授，是Yahoo ACE奖，ICCV Marr奖和ECCV最佳论文奖的获得者。一直协助李飞飞运行**ImageNet** 项目，自2010年以来，协办ImageNet大规模视觉识别挑战赛（ILSVRC）直到2017年。是NIPS 2012和CVPR 2014 BigVision研讨会的主要组织者。

<img src="https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=1508345203,3500768900&fm=26&gp=0.jpg" style="zoom:60%;" />

代表性论文：
![](http://m.qpic.cn/psc?/V50v5bPV1er4BO1DcJVb3iyeFd3aL3dx/ruAMsa53pVQWN7FLK88i5jITwsOckc05dlP9nW8sEgqtgDY1fNcAyVojH4y41EAD5WOu5.SoUXnUEqhOjSHeHz7ib59XjstdNRKhXsEoga8!/b&bo=bQNdAgAAAAADBxM!&rf=viewer_4)

##### 4.1.3 Geoffrey Everest Hinton

加拿大认知心理学家和计算机科学家，被誉为“神经网络之父”、“深度学习鼻祖”、”人工智能教父“。现任Google副总裁兼工程研究员、多伦多大学的特聘教授，也是Vector Institute首席科学顾问。2018年因作为“深度学习领域的三大先驱之一”获得图灵奖，被选为2017年改变全球商业格局的50人。

<img src="https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1605774694262&di=bb2f1207f539a8cb6858663af5d1a641&imgtype=0&src=http%3A%2F%2Fpic2.zhimg.com%2F50%2Fv2-1d49524120dcde04c5f00068cb78b810_hd.jpg" style="zoom:80%;" />

主要贡献：率先将**反向传播(Backpropagation)**用于多层神经网络，发明了**玻尔兹曼机(Boltzmann machine)**，提出**逐层初始化预训练方法**揭开了深度学习的序幕，提出了**胶囊神经网络(capsule network)**。

代表性论文：

1. 反向传播算法的使用

   Rumelhart D E, Hinton G E, Williams R J. Learning representations by  back-propagating errors[J]. Cognitive modeling, 1988, 5(3): 1. 

2. CNN语音识别开篇TDN网络

   Waibel A, Hanazawa T, Hinton G, et al. Phoneme recognition using  time-delay neural networks[J]. Backpropagation: Theory, Architectures  and Applications, 1995: 35-61. 

3. DBN网络的学习 

   Hinton G E, Osindero S, Teh Y W. A fast learning algorithm for deep  belief nets[J]. Neural computation, 2006, 18(7): 1527-1554. 

4. 深度学习的开篇

   Hinton G E, Salakhutdinov R R. Reducing the dimensionality of data  with neural networks[J]. science, 2006, 313(5786): 504-507. 

5. 数据降维可视化方法t-SNE

   Maaten L, Hinton G. Visualizing data using t-SNE[J]. Journal of machine learning research, 2008, 9(Nov): 2579-2605. 

6. DBM模型

   Salakhutdinov R, Hinton G. Deep boltzmann machines[C]//Artificial intelligence and statistics. 2009: 448-455. 

7. **ReLU激活函数的使用**

   Nair V, Hinton G E. Rectified linear units improve restricted  boltzmann machines[C]//Proceedings of the 27th international conference  on machine learning (ICML-10). 2010: 807-814. 

8. RBM模型的训练

   Hinton G E. A practical guide to training restricted Boltzmann  machines[M]//Neural networks: Tricks of the trade. Springer, Berlin,  Heidelberg, 2012: 599-619. 

9. 深度学习语音识别开篇

   Hinton G, Deng L, Yu D, et al. Deep neural networks for acoustic modeling in speech recognition[J]. IEEE Signal  processing magazine, 2012, 29. 

10. **深度学习图像识别开篇AlexNet** 

    Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with  deep convolutional neural networks[C]//Advances in neural information  processing systems. 2012: 1097-1105. 

11. 权重初始化和Momentum优化方法的研究

    Sutskever I, Martens J, Dahl G, et al. On the importance of  initialization and momentum in deep learning[C]//International  conference on machine learning. 2013: 1139-1147. 

12. **Dropout方法提出** 

    Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: a simple way to  prevent neural networks from overfitting[J]. The Journal of Machine  Learning Research, 2014, 15(1): 1929-1958. 

13. 三巨头深度学习综述

    LeCun Y, Bengio Y, Hinton G. Deep learning[J]. nature, 2015, 521(7553): 436. 

14. 蒸馏学习算法

    Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network[J]. arXiv preprint arXiv:1503.02531, 2015. 

15. Capsule NetworkSabour S, Frosst N, Hinton G E.

    Dynamic routing between  capsules[C]//Advances in neural information processing systems. 2017:  3856-3866. 

> 参考：https://www.sohu.com/a/328382912_120233360

##### 4.1.4 Alex Krizhevsky

本文的第一作者，Hinton的博士生。

![](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1605776418461&di=8d8bde39b16b522f57010c020cdfd4e7&imgtype=0&src=http%3A%2F%2Fimages.ofweek.com%2FUpload%2FNews%2F2016-9%2FLiuzhongyang%2F9_16%2F6.jpg)

（从左到右依次为：Ilya Sutskever、Alex Krizhevsky、Geoffrey Hinton）

代表性论文：

![](http://m.qpic.cn/psc?/V50v5bPV1er4BO1DcJVb3iyeFd3aL3dx/ruAMsa53pVQWN7FLK88i5hsSmHqQZw7u5bYU9v9fChL2CP61natzGxjJzAt5LDNedoQsGNRjeOTcAyafdIxY25pCa1l3VZrPVNqyudnhNZg!/b&bo=SANXAgAAAAADBzw!&rf=viewer_4)

##### 4.1.5 Yann LeCun

Hinton的博士后，CNN之父，纽约大学终身教授，前Facebook人工智能研究院负责人，IJCV、PAMI和IEEE Trans 的审稿人，创建了ICLR(International  Conference on Learning Representations)会议并且跟Yoshua Bengio共同担任主席。2014年获得了IEEE神经网络领军人物奖，2018荣获图灵奖。

![](https://ss1.bdstatic.com/70cFvXSh_Q1YnxGkpoWK1HF6hhy/it/u=3425959714,1198945987&fm=15&gp=0.jpg)

主要贡献：1998年开发了**LeNet5**，并制作了被Hinton称为“机器学习界的果蝇”的经典数据集**MNIST**。

代表性论文：

1. 使用反向传播和神经网络识别手写数字

   LeCun Y, Boser B, Denker J S, et al. Backpropagation applied to  handwritten zip code recognition[J]. Neural computation, 1989, 1(4):  541-551. 

2. 早期权值剪枝的研究

   LeCun Y, Denker J S, Solla S A. Optimal brain damage[C]//Advances in neural information processing systems. 1990: 598-605. 

3. 将siamese网络用于签名验证

   Bromley J, Guyon I, LeCun Y, et al. Signature verification using a"  siamese" time delay neural network[C]//Advances in neural information  processing systems. 1994: 737-744. 

4. LeNet5卷积神经网络提出

   LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11):  2278-2324. 

5. 对**max pooling**和average pooling的理论分析

   Boureau Y L, Ponce J, LeCun Y. A theoretical analysis of feature  pooling in visual recognition[C]//Proceedings of the 27th international  conference on machine learning (ICML-10). 2010: 111-118. 

6. DropConnect方法

   Wan L, Zeiler M, Zhang S, et al. Regularization of neural networks  using dropconnect[C]//International conference on machine learning.  2013: 1058-1066. 

7. OverFeat检测框架

   Sermanet P, Eigen D, Zhang X, et al. Overfeat: Integrated recognition, localization and detection using convolutional networks[J]. arXiv  preprint arXiv:1312.6229, 2013. 

8. CNN用于立体匹配

   Zbontar J, LeCun Y. Computing the stereo matching cost with a  convolutional neural network[C]//Proceedings of the IEEE conference on  computer vision and pattern recognition. 2015: 1592-1599. 

9. 三巨头深度学习综述

   LeCun Y, Bengio Y, Hinton G. Deep learning[J]. nature, 2015, 521(7553): 436. 

10. EBGAN

    Zhao J, Mathieu M, LeCun Y. Energy-based generative adversarial network[J]. arXiv preprint arXiv:1609.03126, 2016. 

> 参考：https://www.sohu.com/a/328598636_120233360

#### 4.2 相关的关键工作

##### 4.2.1 随机梯度下降法(SGD)

如果使用梯度下降法，那么每次迭代过程中都要对n个样本进行求梯度，开销非常大。随机梯度下降法是梯度下降法的一种变形，简单但非常有效，基本思想是**随机选取一个样本 $J(x_i)$ 来更新参数**，那么计算开销就从 $O(n)$下降到 $O(1)$。随机梯度下降法多用于支持向量机、逻辑回归等凸损失函数下的线性分类器的学习，并且已成功应用于文本分类和自然语言处理中经常遇到的大规模和稀疏机器学习问题。

随机梯度下降法相较于梯度下降算法的优势可参考：https://zhuanlan.zhihu.com/p/28060786

### 5 文章提出的解决方案的关键

由于ImageNet数据集比以往的图片数据集都大得多，故需要增加网络的深度和宽度来增加网络的容量，因此，这篇文章提出的解决方案中，最关键的就是卷积神经网络的“**深度**”。但是由于深度网络结构的理论基础并不完善，什么数据集需要匹配多深多宽的网络并没有理论依据，需要根据经验和反复的实验来最终确定网络结构。作者在文章中也多次强调，“如果移除任何一个卷积层，网络的性能都会下降。”由此可看出网络结构特别是其深度对系统整体性能的重要性。

另外，本文最大的贡献还在于整合各种技巧：例如ReLU是本文作者Hinton在2010年提出来的改善RBM性能的，把ReLU引入深度CNN中，使得ReLU成为以后深度网络普遍使用的非线性函数，从而在这个领域替代经典的sigmoid、tanh函数；Dropout是Hinton在2012年提出改善深度网络泛化能力；LRN是“What is the best multi-stage architecture for Object Recognition”论文使用的方法，本文简化改进了该归一化方法；而通过纯监督方法学习深度CNN也已经被证明是有效的。最终通过作者的整合得到一个标准的深度CNN网络，ILSVR2013和2014的很多论文都是在结构上修改这个网络，从而获得更好的结果。

对于每一个改进，作者都能将前因后果说得很清楚，并将其效果量化，这种习惯值得学习。由此，我们可以反推出对最终正确率最关键的因素：网络深度（减一层2%、加一层1.7%）> 单层容量（双GPU约1.7%）> LRN（1.4%）> PCA干扰（1%）> 重叠Polling（0.4%）。由此可再次说明，虽然其他的Trick也有明显效果，但决定性的，还是尽可能保持一个足够深、足够大规模的网络。

### 6 实验设计

#### 6.1 模型设计

针对ImageNet这一大型数据集，构建了一个含有6000万个参数、650000神经元的庞大的深度卷积神经网络AlexNet并从提高速度、减少过拟合等方面进行优化。

![](https://img-blog.csdnimg.cn/20190116160802679.png)

![](https://img-blog.csdnimg.cn/20190116160846770.png)

#### 6.2 训练细节

本文使用带动量项的梯度下降法SGD来训练模型。

batch size=128，动量项v=0.9，权值衰减(weight decay) wd=0.0005，W服从均值为0、标准差为0.01的高斯分布。

偏置项：第2、4、5卷积层和全连接层的b=1（促进最初阶段ReLU的学习）；其它层b=0。

学习率：初始为0.01，当验证集的错误率停止降低时，手动缩减学习率（除以10）。

#### 6.3 分析结果

##### 6.3.1 错误率比较

本文的网络在ILSVRC-2010上的结果及对比如下表所示（Tabel 1）。第一行的算法是ILSVRC-2010比赛中的第一名，用的方法是训练六个稀疏编码，然后求这六个模型的预测平均值作为最终结果；第二行的算法是在FVs上训练两个分类器，并求其预测的平均值；第三行即是本文的算法，可以看到本文的算法错误率明显比前两个算法低。

![](https://img-blog.csdnimg.cn/20190529212829355.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA1NjU5MTA=,size_16,color_FFFFFF,t_70)

接着，在ILSVRC-2012上的结果如下表所示（Table 2）。由于当时未公布2012的测试集，所以作者将验证误差当作测试误差，实际上这样的结果并不会差多少，如下表右边第一列就是真正的测试误差，前两列是作者的验证误差。

![](https://img-blog.csdnimg.cn/20190529213926911.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA1NjU5MTA=,size_16,color_FFFFFF,t_70)

**第一行**跟前面的一样是作为对比的算法；

**第二行**使用的就是文中的AlexNet结构的CNN（5个卷积层，3个全连接层），结果top-5是18.2%；

**第三行**是作者用5个类似的AlexNet做预测，然后取它们的平均值，top-5是16.4%（这得花多长时间......）；

**第四行**使用的是在ImageNet Fall 2011数据集上训练一个CNN（结构跟AlexNet类似，但在卷积层的最后新加了一个卷积层，一共6个卷积层，3个全连接层），然后再ILSVRC-2012上fine-tuning，得到的top-5是16.6%（吃力不讨好）；

**第五行**是使用7个CNN，其中两个是在ImageNet Fall 2011数据集上预训练然后fine-tuning的CNN（就是第四行的CNN），另外5个是原本的AlexNet（就是第三行的CNN），将这7个CNN用于预测，然后取其平均值，最终top-5结果是15.4%。

最后，作者将AlexNet用于ImageNet Fall 2009 version的预测。一半数据集用于训练，一半用于测试。最后结果top-1是67.4%，top-5是40.9%，比发布的最好的结果还要好。

##### 6.3.2 定性评估

在下图中，左边部分，作者展示了8张图片的预测结果来说明网络在预测top-5时都从测试图片中学到了什么。右边部分则对比了测试集中的五张图片和在训练集中与之最相似的6张图片，如果两张图片产生的特征激活向量（即CNN的输出结果）的欧几里得距离小，就认为这两张图片相似。但是对于4096维向量的欧几里得距离的计算效率低，所以作者又提出可以训练一个自动编码器来压缩这些向量成二进制编码。

![](https://img-blog.csdnimg.cn/20190529224943580.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTA1NjU5MTA=,size_16,color_FFFFFF,t_70)

### 7 采用的数据集及其是否开源

使用的是ILSVRC大赛中提供的部分ImageNet数据集的子集，其中，ILSVRC-2010的测试集标签是可用的，而ILSVRC-2012的测试集标签不可用。

ImageNet数据集是**开源**的，可在下面的网站下载：

http://www.image-net.org/download-imageurls

### 8 实验结果是否验证了科学假设

是，如下表所示：

| 科学假设                                       | 实验结果                                 |
| :--------------------------------------------- | :--------------------------------------- |
| 用ReLU激活函数代替tanh效果更好                 | 训练速度快了好几倍                       |
| 用两个GPU并行化操作代替一个GPU速度更快         | 错误率降低了1.7%且减少了训练时间         |
| 在某些层后面采用局部响应归一化有助于泛化       | 错误率降低了1.4%                         |
| 采用重叠池化代替传统池化提高准确度             | 错误率降低了0.4%，减少过拟合             |
| 平移图像和水平映射以实现数据增强               | 训练集规模增加了2048倍                   |
| 改变训练图像中RGB通道的灰度以实现数据增强      | top-1错误率降低了1%以上                  |
| 采用Dropout正则化可以减少过拟合                | 可以防止过拟合但收敛的迭代次数可能会翻倍 |
| 综合以上改进，这个深度卷积神经网络性能大大提高 | 测试结果比之前的神经网络都要好           |

### 9 本文贡献

本文的主要贡献在于针对ImageNet这样一个较大较复杂的数据集，提出并搭建一个大型且复杂的深度卷积神经网络**AlexNet**来更好地实现识别和分类任务，并且突破了当时的记录，有效地解决了过拟合的问题。

本文最大的贡献还在于整合各种技巧，对CNN的一些改进成为以后CNN网络通用的结构，之后在ImageNet上取得更好结果的ZF-net、SPP-net、VGG等网络，都是在其基础上修改得到的。

**AlexNet**在深度学习发展史上的**历史意义远大于其模型的影响**。在此之前，深度学习已经沉寂了很久。在此之后，深度学习重新迎来春天，卷积神经网络也成为计算机视觉的核心算法模型。

> **如果我们今天回过头看看，将人工智能领域的蓬勃发展归功于某个事件的话，这份殊荣应属于2012年 ImageNet大赛的比赛成果。**
>
> **2012年 ImageNet 的那场赛事的的确确引发了今天人工智能井喷式的发展。之前在语音识别领域是有一些成果，但大众并不知道，也不关心，而 ImageNet 让人工智能开始进入公众视野。**

### 10 下一步怎么做（本文局限性）

到目前为止，本文的结果已经获得了足够的进步，因为已经使网络更大，并且训练了更长时间。但仍然有很大的空间去优化网络，使之能够像人类的视觉系统一样感知。最后，作者希望接下来能够对**视频序列**也使用这样非常大的深度卷积神经网路，因为视频序列的时间结构提供了非常有用的信息，而这些信息往往在静态图像中丢失了或者很不明显。

### 11 重要的相关论文

1. D. Cire¸san, U. Meier, and J. Schmidhuber. Multi-column deep neural networks for image classification. Arxiv preprint arXiv:1202.2745, 2012.

   在文中被引用了3次。

2. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09, 2009.

   首次公开ImageNet数据集。

3. G.E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R.R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. arXiv preprint arXiv:1207.0580, 2012.

   提出Dropout正则化方法。

4. K. Jarrett, K. Kavukcuoglu, M. A. Ranzato, and Y. LeCun. What is the best multi-stage architecture for object recognition? In International Conference on Computer Vision, pages 2146–2153. IEEE, 2009.

   在文中被引用了4次，提出局部响应归一化方法。

5. V. Nair and G. E. Hinton. Rectified linear units improve restricted boltzmann machines. In Proc. 27th International Conference on Machine Learning, 2010.

   提出ReLU激活函数。

6. Karen Simonyan et al. Very Deep Convolutional Network for Large-scale Image Recognition. ICLR, 2015.

   在AlexNet的基础上改进提出VGG网络架构。

7. Christian Szegedy et al. Going Deeper with Convolutions. CVPR, 2015. 

   在ILSVRC 2014于AlexNet的基础上改进，提出参数比其少12倍但准确性大大提高的GoogLeNet网络架构。

8. Kaiming He et al. Deep Residual Learning for Image Recognition. CVPR, 2015.

   在AlexNet的基础上改进提出ResNet网络架构。

### 12 不懂之处

1. 对使用两个GPU进行实验的方法比较模糊，以及对最后做的定性分析不是很清楚具体的方式和步骤。
2. 在result中使用5个、7个CNN进行预测具体是如何操作？

### 13 专业术语

举个例子，比如训练好了一个网络，要用这个网络去进行图片分类任务，假设要分类的数目有50类，那么在进行测试时，输入一张图片，网络会依次输出这50个类别的概率，当所有图片测试完成后，那么：**TOP-5正确率**就是说，在测试图片的50个分类概率中，取前面5个最大的分类概率，里面有没有包含正确的标签，如果有，就分类成功，那么TOP-5正确率此时等于：所有测试图片中正确标签在前五个分类概率的个数/所有的测试图片数。

#### 13.1 top-5错误率

TOP-5错误率比较实际类别与前5个预测类别，即正确标记的样本不在前五个概率里面的样本数除以总的样本数。

#### 13.2 top-1错误率

TOP-1错误率比较实际类别和第一个预测类别，即正确标记的样本不是**最佳概率**的样本数除以总的样本数。

> 参考：https://blog.csdn.net/qq_26413875/article/details/100542817

### 14 英语词汇

**high-resolution**：高分辨率

**top-1 error rate**：top1错误率，即分类预测的最大可能结果不是正确标签的概率

**top-5 error rate**：top5错误率，即分类预测的前5个可能结果都不是正确标签的概率

**neuron**：神经元

**non-saturating**：非饱和的

**object recognition**：目标识别/对象检测

**label-preserving transformations**：标签保留转换，一种数据增强从而减少过拟合的方式

**feedforward neural network**：前馈神经网络

**down-sampled**：下采样

**rescaled**：重新调整图像大小

**nonlinearity**：非线性单元

**gradient descent**：梯度下降

**stochastic gradient descent(SGD)**：随机梯度下降法

**Rectified Linear Units(ReLUs)**：修正线性单元

**iteration**：迭代

**cross-GPU parallelization**：跨CPU并行化操作

**local response normalization**：局部响应归一化

**generalization**：泛化

**activity**：激活值

**overlapping pooling**：重叠池化，即移动步长小于核尺寸

**data augmentation**：数据增强

**weight decay**：权重衰减

**fine-tuning**：微调(网络)

**Euclidean distance**：欧几里得距离/欧氏距离

> 译文参考：https://blog.csdn.net/hongbin_xu/article/details/80271291
>
> 文献解读参考：
>
> 1. https://blog.csdn.net/zzc15806/article/details/83420899?utm_medium=distribute.pc_relevant.none-task-blog-title-2&spm=1001.2101.3001.4242
> 2. https://www.cnblogs.com/zhang-yd/p/5808283.html
> 3. https://blog.csdn.net/whiteinblue/article/details/43202399?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf
> 4. https://zhuanlan.zhihu.com/p/69273192?utm_source=qq
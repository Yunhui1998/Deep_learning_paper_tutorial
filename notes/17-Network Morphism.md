## Network Morphism

### 1  论文的结构(简要概括)

#### 1.1  Abstract

- 作者想解决什么问题？

  如何将训练好的模型变形到新的模型，保留模型函数，缩短训练时间。

- 作者通过什么理论/模型来解决这个问题？

  通过网络态射（Network Morphism）的方法，针对网络态射的两种要求提出解决方法：

  1、处理不同的网络形态类型；

  方法:引入网络的射态方程，提出新的射态变形算法；
  2、处理网络中的非线性；

  方法：引入了参数激活函数族的概念，将非线性操作简化为具体可学习参数的线性操作。

- 作者给出的答案是什么？

  通过网络态射，使得使新网络继承原网络的知识，缩短训练时间，在此基础上进一步提升。

#### 1.2 Introduction

- 作者为什么研究这个课题？

  深度卷积神经网络（DCNN）很火，但训练网络的时间很长。希望可以让训练好的网络只需要花很小的代价就可以直接用于其他相关应用程序。

- 目前这个课题的研究进行到了哪一阶段？存在哪些缺陷？作者是想通过本文解决哪个问题？

  目前这个课题的研究有IdMorph，但由于IdMorph的识别层稀疏性问题，有时候会失败，因此作者引入一些态射操作和提出基于反卷积的网络深度态射方法来解决这个问题。同时，这个课题还面临非线性操作问题，对于非线性问题，无法直接通过线性方法解决，作者引入了参数激活函数族的概念，可将非线性操作简化为具体可学习参数的线性操作。

- **Network Morphism的介绍**

  父网络->子网络的变化:深度变形：s->s+t;

  宽度变形和核大小变化：r
  CD段：嵌入了一个子网络

  <img src="https://ae02.alicdn.com/kf/U0c8565571cca46a089bced060f561282M.png" alt="image.png" title="image.png" />

#### 1.3 Related work

- 和作者这篇论文相关的工作及优缺点有哪些？

  * Mimic Learning
  * Pre-training and Transfer Learning
  * Net2Net

- 作者主要是对之前的哪个工作进行改进？

  作者主要是对IdMorph进行改进。

#### 1.4  Theoretical Analysis

- 作者是用什么理论证明了自己的方法在理论上也是有保障的？

  Network Morphism算法，主要包括线性和非线性算法。

#### 1.5 Experiment

- 作者是在哪些数据集或者说场景下进行了测试？

  MNIST, CIFAR10, and ImageNet

- 实验中的重要指标有哪些？

  准确率、Top-1、Top-5。

- 文章提出的方法在哪些指标上表现好？

  文章通过实验展示了所提出的方法在以下三个方面的有效性：

  ​	1)不同的变形运算；

  ​	2)经典网络和卷积的神经网络；

  ​	3)幂等激活(ReLU)和非幂等激活(TanH)。

#### 1.6 Conclusion

- 这篇论文最大的贡献是什么？

  系统了介绍了网络态射方法，引入网络的射态方程，提出新的射态变形算法，可以将一个训练好的父网络在保留网络函数的同时，变形到一个新的子网络，而子网络在此基础上可以继续训练成更好的网络。

### 2 论文想要解决的问题？

#### 2.1 背景是什么？

深度卷积神经网络（DCNN）很火，但训练网络的时间很长。希望可以让训练好的网络只需要花很小的代价就可以直接用于其他相关应用程序。

#### 2.2 之前的方法存在哪些问题

目前这个课题的研究有IdMorph，但由于IdMorph的识别层稀疏性问题，有时候会失败，因此作者引入一些态射操作和提出基于反卷积的网络深度态射方法来解决这个问题。同时，这个课题还面临非线性操作问题，对于非线性问题，无法直接通过线性方法解决，作者引入了参数激活函数族的概念，可将非线性操作简化为具体可学习参数的线性操作。

### 3 论文研究的是否是一个新问题

不是，研究的是如何缩短网络训练时间。

### 4 论文试图验证的科学假设

#### 4.1 Network Morphism: Linear Case

去掉所有的非线性激活函数，考虑一个只连接完全连接层的神经网络。

原理图：

<img src="https://ae04.alicdn.com/kf/H30f99c8a7e3d4a3d8743dbe0ec8ab7326.png" alt="image.png" title="image.png" />

​		

最终得到的公式：

<img src="https://ae03.alicdn.com/kf/H871ae693df5f400983c6abb2423ba124H.png" alt="image.png" title="image.png" />

网络深度变形输入输出：

<img src="https://ae02.alicdn.com/kf/H22bb1cda372d4e6ebbba613bd82a47b7O.png" alt="image.png" title="image.png" />



<img src="https://ae01.alicdn.com/kf/H72ce13cb0f184460993893bbf3a43eade.png" alt="image.png" title="image.png" />



#### 4.2 Network Morphism Algorithms: Linear Case

​	设计了两个算法解决公式(6),采用反卷积方式，第一种在一定条件下用非0元素填充所有参数；第二种不依赖该条件，渐进地用非0元素填充

##### 4.2.1 GENERAL NETWORK MORPHISM

<img src="https://ae04.alicdn.com/kf/H7ad4405a67ab4f87be45967acfc37f9bS.png" alt="image.png" title="image.png" />

提出了在一定条件下求解式(6)的算法。如算法1所示，用随机噪声初始化子网络的Fl和Fl+1卷积核。然后通过修正另一个迭代求解Fl+1和Fl。对于每次迭代，Fl+1或Fl通过反卷积求解。因此，总损失总是在减少，并有望收敛。然而，不能保证算法1中的损失总是收敛到0。我们认为，如果Fl+1或Fl的参数个数不小于G~，则算法1收敛于0。权利要求1。如果满足下列条件，则损失在

算法1收敛到0(一步)：

<img src="https://ae03.alicdn.com/kf/H5822a96d97ff4b0383987900b0dfb0d46.png" alt="image.png" title="image.png" />

条件(7)中的三项分别为Fl+1、Fl和G~的参数。由于随机矩阵不一致的存在(概率为0)，所以不确定线性系统的解总是存在的。

##### 4.2.2 PRACTICAL NETWORK MORPHISM

<img src="https://ae05.alicdn.com/kf/H73866b4bb3be4e2589a29f0c3d018280s.png" alt="image.png" title="image.png" />

算法2的非稀疏实践的牺牲如图3所示。算法2用非稀疏牺牲消除条件(7)。

(a)为IdMorph填充结果；(b)为态射网络最坏情况；(c)为态射网络最好情况；大多数情况都介于(b)和(c)之间

<img src="https://ae02.alicdn.com/kf/H3cf2cdb285cc4b18ba4e09848e402b9bs.png" alt="image.png" title="image.png" />

##### 4.2.3 对比

<img src="https://ae02.alicdn.com/kf/H3e15f1aefa344f528a62414109b3b791w.png" alt="image.png" title="image.png" />

#### 4.3 Network Morphism: Non-linear Case

<img src="https://ae01.alicdn.com/kf/H168e01e11d47402e8575250d7eb18a25Q.png" alt="image.png" title="image.png" />

如图所示，添加绿色框所示的非线性激活是安全的，但是我们需要确保黄色框最初等同于线性激活。在训练的时候，需要学习a的值。一旦学会了a的值，这种线性激活将转换成为非线性激活。

传统的：

<img src="https://ae03.alicdn.com/kf/Hd1b087313c90410eb759bf7c8e8cc3b4f.png" alt="image.png" title="image.png" />

Network Morphism改进后：

<img src="https://ae02.alicdn.com/kf/H51f402f9de2046a18683d507a66a835bZ.png" alt="image.png" title="image.png" />

#### 4.4 Stand-aloneWidth and Kernel Size Morphing

##### 4.4.1 WIDTH MORPHING

<img src="https://ae05.alicdn.com/kf/U5f4c45139cb94ab487bbc922471d949aM.png" alt="image.png" title="image.png" />

​	设置少参数的那个为0，另外一个设置成随机噪声。

<img src="https://ae05.alicdn.com/kf/H2fd0a17f23e94f3da148656fb756b1caO.png" alt="image.png" title="image.png" />

##### 4.4.2 KERNEL SIZE MORPHING

针对核尺寸变形问题，提出了一种启发式的有效解决方案。假设卷积层l的内核大小是Kl，我们想把它扩展到K~l。当第l层的过滤器两边填充![img](https://img-blog.csdnimg.cn/20190308162248144.png)个0时，同样的操作也适用于blobs。如图5所示，得到的blobs形状相同，数值也相同。

<img src="https://ae02.alicdn.com/kf/H65e8ce3ea8194497b2c33ae9e43dd71aG.png" alt="image.png" title="image.png" />

#### 4.5 Subnet Morphing

一种策略是首先设计子网模板，然后通过这些子网构建网络。两个典型的例子是Network in Network (NiN)的mlpconv层(Lin et al.， 2013)和inception层GoogLeNet (Szegedy et al.， 2014)，如图6(a)所示。顺序子网变形是指从单一层向多个顺序层进行变形，如图6(b)所示。对于堆叠顺序子网变形，可以按照如图6(c)所示的工作流程。首先，父网络中的一层被分割成多条路径，其中满足条件，

<img src="https://ae05.alicdn.com/kf/Uf9f9b3e233c9468c980ef925f6cf8e05Q.png" alt="image.png" title="image.png" />

然后，对每条路径进行顺序的子网变形。

<img src="https://ae05.alicdn.com/kf/Hc2bfb825545a4d6487da0d60397561c0p.png" alt="image.png" title="image.png" />

### 5 相关的关键人物与工作

#### 5.1 之前存在哪些相关的工作

* Mimic Learning

  Mimic Learning即为镜像学习，类似于一个网络作为学生，学习另一个网络的知识。与之不同的是，网络态射（Network Morphism）并非从0开始学习，而是类似于父子关系，子网络直接从父网络种继承完整的知识、网络函数。

* Pre-training and Transfer Learning

  预训练和迁移学习都改变了父网络最后几层的参数同时冻结其他层，预训练是在同一个数据集下对子网络进行训练，而迁移学习则是在一个新的数据集中进行训练，可以理解为，预训练中子网络和父网络解决的是同一个问题，而迁移学习则是解决新的问题。由于这两个方法改变了最后几层的参数，因此网络函数同时也被改变。

* Net2Net

#### 5.2 本文是对哪个工作进行的改进

Net2Net中用到的IdMorph，其改进的部分有：

1、方法不同；Net2Net局限于IdMorph方法，而网络态射（Network Morphism）可以使网络可以嵌入非识别层；
2、Net2Net只对幂等激活函数有效，态射网络可以处理非线性激活函数；
3、Net2Net只讨论了宽度和深度，态射网络研究了深度、宽度、内核大小和子网变化等；
4、Net2Net只能分别考虑深度和宽度，而态射网络可以同时考虑深度、宽度和内核大小变化。

#### 5.3 这个领域的关键研究者

##### 5.3.1 Tianqi, Chen

​	陈天奇，华盛顿大学计算机系博士生，研究方向为大规模机器学习。他曾获得 KDD CUP 2012 Track 1 第一名，并开发了 SVDFeature，XGBoost，cxxnet 等著名机器学习工具，是 [Distributed (Deep) Machine Learning Common](https://github.com/dmlc/) 的发起人之一。

##### 															<img src="https://ae05.alicdn.com/kf/U0c7fe0e779cf4ce0a6b5127da801eb640.png" alt="image.png" title="image.png" />

<img src="https://ae04.alicdn.com/kf/U478061e85937475d93363b1cdad5aa65k.png" alt="image.png" title="image.png" />

##### 5.3.2 Ian J. Goodfellow

​	Ian *Goodfellow*因提出了生成对抗网络（GANs）而闻名，他被誉为“GANs之父”，甚至被推举为人工智能领域的顶级专家。

<img src="https://ae03.alicdn.com/kf/U640040481f0a4cacb55791083d715b2aW.png" alt="image.png" title="image.png" />

<img src="https://ae03.alicdn.com/kf/Ue7201d7378ef4782ae4e7a7c03449648l.png" alt="image.png" title="image.png" />

##### 5.3.3 Jonathon Shlens

​	PCA发明者。

​																	<img src="https://ae05.alicdn.com/kf/Uf80e684ecfd140a68a65fdaa7301772eh.png" alt="image.png" title="image.png" />
<img src="https://ae01.alicdn.com/kf/U9cca0bed8b294bf2bb41859599998d9eB.png" alt="image.png" title="image.png" />

### 6 论文提出的解决方案的关键

​	引入网络的射态方程，提出新的射态变形算法；同时提出一系列激活函数。

### 7 论文的解决方案有完备的理论证明吗

​	有。见论文试图验证的科学假设。

### 8 实验设计

#### 8.1用到了哪些数据集

​	MNIST, CIFAR10, and ImageNet

#### 8.2与什么算法进行了比较

##### 8.2.1 Network Morphism for Classic Neural Networks

​	

<img src="https://ae05.alicdn.com/kf/Hae477379c4e64bdb92c1009fb931f8a2I.png" alt="image.png" title="image.png" />

##### 8.2.2 Depth Morphing, Subnet Morphing, and Internal Regularization for DCNN

​	

<img src="https://ae02.alicdn.com/kf/H8930e48c6380496e900b4bbc2428eedcA.png" alt="image.png" title="image.png" />

##### 8.2.3 Kernel Size Morphing and Width Morphing



<img src="https://ae03.alicdn.com/kf/Hc570a071a3eb4b639a03d9af2cf4c12c8.png" alt="image.png" title="image.png" />

<img src="https://ae04.alicdn.com/kf/H068fdb06407d4ff7ac19548d08eb9fca1.png" alt="image.png" title="image.png" />

​		VGG16(NetMorph)在单GPU上训练需要5天，而VGG16需要2-3个月，提升速度15x。

#### 8.3评价指标是什么

​	准确率、Top-1、Top-5。

### 9 实验支撑

#### 9.1 论文的数据集哪里获取

​	MNIST：http://yann.lecun.com/exdb/mnist/

​	CIFAR10：http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

​	ImageNet：http://www.image-net.org/ 

#### 9.2 源代码哪里可以获取

​	关于本文的代码未找到，但找了文中所提及的Net2Net，以及之后网络态射相关的AutoKeras。

Net2Net：https://github.com/soumith/net2net.torch

2018 **Efficient Neural Architecture Search with Network Morphism**：https://github.com/keras-team/autokeras

#### 9.3 关键代码的讲解

​	网络态射的一个比较火的开源应用是autokeras，其网络结构用 keras 的图模型 graph 表示。

​	graph 类中的每个节点都是层之间的中间张量，每一层都是图中的一条边。

​	graph 类中包含所有节点 (包括它们的 shape 和 id)、所有的层（包含 层 本身和它们的 id）、关系（层和输入节点、输出节点的关系以及邻接矩阵）。

```python
# in autokeras/graph.py
def _build_network(self):
    self._node_to_id = {}
	# Recursively find all the interested nodes.
    for input_node in self.inputs:
        self._search_network(input_node, self.outputs, set(), set())
    self._nodes = sorted(
        list(self._node_to_id.keys()), key=lambda x: self._node_to_id[x]
    )

    for node in self.inputs + self.outputs:
        if node not in self._node_to_id:
            raise ValueError("Inputs and outputs not connected.")

    # Find the blocks.
    blocks = []
    for input_node in self._nodes:
        for block in input_node.out_blocks:
            if (
                any(
                    [
                        output_node in self._node_to_id
                        for output_node in block.outputs
                    ]
                )
                and block not in blocks
            ):
                blocks.append(block)

    # Check if all the inputs of the blocks are set as inputs.
    for block in blocks:
        for input_node in block.inputs:
            if input_node not in self._node_to_id:
                raise ValueError(
                    "A required input is missing for HyperModel "
                    "{name}.".format(name=block.name)
                )

    # Calculate the in degree of all the nodes
    in_degree = [0] * len(self._nodes)
    for node_id, node in enumerate(self._nodes):
        in_degree[node_id] = len(
            [block for block in node.in_blocks if block in blocks]
        )

    # Add the blocks in topological order.
    self.blocks = []
    self._block_to_id = {}
    while len(blocks) != 0:
        new_added = []

        # Collect blocks with in degree 0.
        for block in blocks:
            if any([in_degree[self._node_to_id[node]] for node in block.inputs]):
                continue
            new_added.append(block)

        # Remove the collected blocks from blocks.
        for block in new_added:
            blocks.remove(block)

        for block in new_added:
            # Add the collected blocks to the Graph.
            self._add_block(block)

            # Decrease the in degree of the output nodes.
            for output_node in block.outputs:
                output_node_id = self._node_to_id[output_node]
                in_degree[output_node_id] -= 1`
```
### 10 实验结果是否验证了科学假设？

​	验证了。

### 11 论文最大的贡献

​	提出了网络态射（Network Morphism）的方法解决如何将训练好的模型变形到新的模型，保留模型函数，缩短训练时间。

### 12 论文的不足之处

#### 12.1 这篇论文之后的工作有哪些其他的改进

​	AutoKeras：https://autokeras.com/

​	利用了神经架构搜索，但应用的是"网络态射"（保持网络功能而改变网络结构）以及贝叶斯优化。用来引导网络态射实现更有效的神经网络搜索。https://arxiv.org/abs/1806.10282

#### 12.2你觉得可以对这篇论文有什么改进

* 不改变父网络的函数，即意味着只能在同一个问题上进行修改，可以改进其适用于多个问题。
* 利用网络态射自动化生成模型，优化其自动化生成算法。

### 13 重要的相关论文

Chen, Tianqi, Goodfellow, Ian, and Shlens, Jonathon.Net2net: Accelerating learning via knowledge transfer.arXiv preprint arXiv:1511.05641, 2015.

本文主要是基于这篇文章的IdMorph进行改进。

### 14 不懂之处

对于其中的部分理论公式推导搞不太懂。

<img src="https://ae03.alicdn.com/kf/Hff379d3e03e84a76b6e8b8f7b6f3ddaes.png" alt="image.png" title="image.png" />

<img src="https://ae04.alicdn.com/kf/H115333bb6258423ba4b0f6e138fa7c3cF.png" alt="image.png" title="image.png" />





### 15 其他

**（2019.11-Self-training with Noisy Student improves ImageNet classification）**

文章首先在标注的 ImageNet 图像上训练了一个 EfficientNet 模型，然后用这个模型作为老师在 3 亿无标签图像上生成伪标签。然后研究者训练了一个更大的 EfficientNet 作为学生模型，使用的数据则是正确标注图像和伪标注图像的混合数据。

这一过程不断迭代，每个新的学生模型作为下一轮的老师模型，在生成伪标签的过程中，教师模型不会被噪声干扰，所以生成的伪标注会尽可能逼真。但是在学生模型训练的过程中，研究者对数据加入了噪声，使用了诸如数据增强、dropout、随机深度等方法，使得学生模型在从伪标签训练的过程中更加艰难。

这一自训练模型，能够在 ImageNet 上达到 87.4% 的 top-1 精确度，这一结果比当前的 SOTA 模型表现提高了一个点。除此之外，该模型在 ImageNet 鲁棒性测试集上有更好的效果，它相比之前的 SOTA 模型能应对更多特殊情况。

### 16 引用本文的重要论文

#### Auto-Keras: An Efficient Neural Architecture Search System

##### 16.1 链接：

​		https://autokeras.com/

##### 16.2 算法流程

<img src="https://ae05.alicdn.com/kf/H0aa14d7244c34833a91e399a3d61f23bv.png" alt="image.png" title="image.png" />

##### 16.3 初始化模型

程序初始化会将网络模块放入生成器作为种子。网络模块包括 MlpModule 和 CnnModule。

CnnModule 的generators是个列表，包括了三个生成器: CnnGenerator,ResNetGenerator 和DenseNetGenerator分别用于生成一般的CNN，ResNe和DenseNet。CnnGenerator 包含了 conv, dropout, global-avg-pooling,pooling, batch_norm 等层结构，初始默认生成三层卷积层的网络结构。如下图所示，初始时生成三个对应的model，分别为CNN, ResNet, DenseNet。

<img src="https://ae04.alicdn.com/kf/H725272ebd30441d287a973ca91552e31T.png" alt="image.png" title="image.png" />

##### 16.4 训练

程序会将所有的model放进training queue(训练队列）中，然后开始弹出model并且训练，得到model在相应数据集中的评分，这个训练只为了评估，因此不需要充分训练以节省资源。与此同时进行网络结构的搜索。

##### 16.5 搜索

在搜索的过程中，对于model和它们训练得到的评估进行排序放入搜索队列，在时间允许的范围内，循环下面过程：

对队列中的model（先进先出，先出的是评估好的）采用退火算法判断是（exploit) 否 (explore) 要对网络结构进行 Morphism。假如产生了变形且变形结果并不在已有的model里面，用高斯过程回归估计它的acq分数，

在循环过程中记录acq最高的网络，最后返回这个网络加入model list中和训练队列中。程序训练这个model得到对应的评分，然后更新数据。

训练和搜索是同时进行的，双进程进行。

而 Network Morphism在 autokeras 中提供了以下几种变形：深度、宽度和层之间的连接。Morphism 是随机的，例如选择哪种变形方式是随机的。当选择增加宽度时，选取加宽哪一层也是随机的。
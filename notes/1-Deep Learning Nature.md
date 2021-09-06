## 1 Deep Learning Nature

### 1 文章想要解决的问题

本论文是一篇综述，解决的问题是概括现在的深度学习的发展，并且给出未来可行的方向。

### 2 研究的是否是一个新问题

不是。深度学习的未来应该怎么发展、应该朝着哪一个方向研究，应该不算新问题。但毫无疑问是一个重要的问题，因为这决定了深度学习当前的发展速度以及所能到达的水平。

### 3 相关的关键人物与工作

#### 3.1 相关的关键工作

相关的关键工作包括监督学习、反向传播、卷积神经网络、深度卷积网络的图像理解、词向量（Distributed representations）和语言处理以及循环神经网络。

#### 3.2 相关的关键人物

关键人物：深度学习三巨头（同时也是这篇论文的作者们）**Geoffrey Hinton**、**Yann LeCun**、**Yoshua Bengio**以及LSTM之父**Jürgen Schmidhuber**。论文中缩写分别为Hinton、Le、Bengio和 Schmidhuber。

### 4 本文贡献

该论文最后展望了深度学习的未来。首先是虽然现在是监督学习更好，但是真正的智能应该是无监督的。然后在计算机视觉方面，希望使用强化学习决定放注意力到图片的哪个地方，并结合CNN、RNN等技术开发端到端的系统；在自然语言理解方面，希望拥有任意时刻都能选择性处理文本某部分的策略。

以及提到人工智能领域的未来进展来自将表征学习（representation learning）和 复杂推理（complex reasoning）结合起来的系统。

其中表征学习/表示学习在花书第十五章，大致是学习如何将数据转换成另一种表示（比如图片从RGB转化为HSV），比较典型的是自编码器（self-encoding）；Reasoning还没找到很好的解释，暂不提。不过这句话引自《From machine learning to machine reasoning》，或许会更加理解这个的含义。

### 5 下一步怎么做（本文局限性）

论文给出了一定的方向，其中可能最容易实现并且有不错效果的是将RNN、CNN以及强化学习联合一起使用。最终目标应该是实现无监督学习且效果媲美现在的SOTA的算法。

### 6 重要的相关论文

由于是综述，所以其引用的大部分论文都可以认为是重要的相关论文。并且在原论文的应用处，还会对部分影响力特别大的论文给出评价（这些论文都是三巨头的，只有一篇来自LSTM之父）。

除去云辉师兄发的8篇，包括 《[Deep neural networks for acoustic modeling in speech recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6296526)》、《[Sequence to sequence learning with neural networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-)》、《[Deep sparse rectififier neural networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)》、《[Greedy layer-wise training of deep networks](https://proceedings.neurips.cc/paper/2006/file/5da713a690c067105aeb2fae32403405-Paper.pdf)》、《[Handwritten digit recognition with a back-propagation network](http://papers.nips.cc/paper/293-handwritten-digit-recognition-with-a-back-propagation-network.pdf)》、《[Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/abstract/document/726791)》、《[A neural probabilistic language model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)》、《[Long short-term memory](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)》。

### 7 专业术语

Deep learning：深度学习

Representation learning：表征学习

Supervised learning：监督学习

Unsupervised learning：无监督学习

Objective function：目标函数，在大部分情况下等同于损失函数的意义

Backpropagation：反向传播

chain rule for derivatives：链式求导，chain rule表示链式法则

Convolutional neural networks：卷积神经网络

### 8 英语词汇

review：总结；综述
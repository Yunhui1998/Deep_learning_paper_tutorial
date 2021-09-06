## 24-Deep Speech 2_ End-to-End Speech Recognition in English and Mandarin

深度语音2:中英文的端到端语音识别

### 1  论文的结构

#### 1.1  Abstract

- 作者想解决什么问题？

  构建端到端的英文及普通话的语音识别系统，目标是最终达到人类级别的性能，且可以在噪音、口音、多语言等情况下通用。

- 作者通过什么理论/模型来解决这个问题？

  尝试了很多网络结构和技术：使用SortaGrad、Batch Nomalization、CTC、BRNN和GRN，1D/2D invariant convolution来增强数值优化，以提升语音识别的精度。

  使用数据合成来进步增强数据。

  提升训练速度：在一个经过优化的、高性能计算(HPC, High-performance Computing)系统上实现，能够在短短几天之内对大型数据集进行全量的训练。

#### 1.2 Introduction

- 作者为什么研究这个课题？

  随着数据及算力的持续增加，对于语音识别系统，端到端的深度学习已经进入到目前最先进的自动语音识别（ASR，automatic speech recognition）。

- 和作者这篇论文相关的工作有哪些？

  关于深度学习中端到端语音识别和模型训练的可伸缩性优化。

  1. 针对语音识别网络模型，前馈神经网络FNN的声学模型早在二十多年前就提出了，当时RNN和CNN也被应用在了语音识别上。之后又提出了RNN+LSTM,以及探索了双向循环网络BRNNs的模型。

  2. 关于将音频序列转录为输出，结合注意力机制的RNN编码解码器在预测音素或字素时表现很好。之后又提出CTC损失函数 + RNN，在按字符输出的端到端的语音识别系统上效果显著，不需要在训练前进行帧对齐。

  3. 提升GPU效率，对低层网络有用，而模型并行+数据并行[64，26]可以用于创建一个高速的、可伸缩的系统，用于深度RNN网络训练语音识别。

  4. 数据增强在CV及ASR领域都非常有效。现有的语音识别系统也可以引导新数据生成，比如可以用于对齐及过滤数千小时的语音。

#### 1.5 Experiment

- 作者是在哪些数据集或者说场景下进行了测试？

  相对于人工语音识别和DS1所做的算法改进对比

  ldc语料库中的华尔街日报(WSJ)语料

  LibriSpeech语料库

  其他数据集为内部百度语料库。

- 实验中的重要指标有哪些？

  英语系统的单词错误率(WER)：插入、替换或删除的词的总个数，除以标准的词序列中词的总个数的百分比
  
  汉语系统的字符错误率(CER)
  
  $$
  \text { WER }=100 \cdot \frac{S+D+I}{N} \% \quad \text { Accuracy }=100-\text { WER\% }
  $$

#### 1.6 Conclusion

- 这篇论文最大的贡献是什么？

  1. 模型结构
  2. 大量标记的训练集
  3. 语音识别的计算规模

- 论文中的方法还存在什么问题？

  在真实嘈杂环境的数据上，DS2语音识别和人类水平依然存在差距。

- 作者觉得还可以怎么改进？

  对于具体的应用场景，对转录可能有特殊的要求，这些需要做后处理（不直接输出字符，例如数字格式规范化）。需要进一步探究其适应性和应用场景。

### 2 论文想要解决的问题？

#### 2.1 背景是什么？

在DS1基础上，使用一系列深度学习技术，实现高精度，高泛化功能，适用于现实中复杂情况的语音识别环境。

#### 2.2 之前的方法存在哪些问题

整合之前的方法，提升精度和泛化能力。

#### 2.3 输入和输出是什么？

网络的输入是一个指数归一化的语音片段的对数频谱序列，窗口大小为20ms。输出是某种语言的一些字符。

在英文中，lt属于{a,b,c…,z, 间隙（space）,撇号(apostrophe),空白（blank）}，我间隙代表每个单词的边界。对于普通话，网络的输出是简体中文字符，有6000个字符，其中包括罗马字母。

### 3 论文研究的是否是一个新问题

不是，端到端的学习之前已提出：高泛化能力，更换识别语言体系时可以利用相同的框架结构直接训练，无需对齐和字典等等。本文力图构建一个网络模型，充分发挥端到端技术的优点。

### 4 论文试图验证的科学假设

DS2对英文和普通话的语音识别精度可以接近甚至超过人工识别，能够应用在嘈杂环境下，且降低网络训练时长。

### 5 相关的关键人物与工作

#### 5.1 之前存在哪些相关的工作

1.2 Introduction中的相关技术。

#### 5.2 这个领域的关键研究者

百度于 2015 年 11 月发布的 Deep Speech 2 已经能够达到 97% 的准确率，并被麻省理工科技评论评为 2016 年十大技术突破之一。

### 6 论文提出的解决方案的关键

#### 6.1 模型结构

![img](https://img2020.cnblogs.com/blog/802043/202008/802043-20200824141902787-767480021.png)

一个循环神经网络（RNN），输入首先是3个 Conv layer，后面接了7个GRU循环层或者双向循环层。然后接了一个全连接层，输出为softmax层输出。网络使用的是CTC损失函数进行端到端的训练，CTC直接从输入音频中预测符号序列。

CTC模型是用一个更大的文本集合进行训练的，使用特别的定向搜索（beam search）寻找转录 y ,使得下式最大化：

$$
Q(y)=\log \left(p_{\mathrm{RNN}}(y \mid x)\right)+\alpha \log \left(p_{\mathrm{LM}}(y)\right)+\beta \mathrm{wc}(y)
$$
wc(y)是转录y中英文或者中文的数量，α控制着语言模型和CTC网络的相对贡献。权重β鼓励转录出更多的字。

模型训练用的是SGD+ Nesterov momentum，mini batch = 512，学习率取[10^-4 , 6*10^-4]，每次迭代后以常数因子1.2进行退火。Momentum设置的是0.99。


#### 6.2 batch normalization（批归一化）

Batch normalization只加在RNN的垂直连接上，来更快地训练深度网络。

BRNN计算公式：
$$
h_{t}^{l}=f\left(W^{l} h_{t}^{l-1}+U^{l} h_{t-1}^{l}+b\right)
$$
即当前 $l$ 层 时间步 t 的激活值是通过将同一时间步 t 的前一层（$l$−1）的激活值与在当前层的前一个时间步 t-1 的激活值相结合来计算的。关于BRNN，每一层的 $h_t$ 使用两个独立的隐藏层来处理两个方向的数据，然后将数据转发到相同的输出层：
$$
\begin{aligned}
\vec{h}^l_{t} &=\mathcal{H}\left(W_{x} \vec{h}^{l-1}_{t}+W_{\vec{h}} \vec{h} \vec{h}^l_{t-1}+b_{\vec{h}}\right) \\
\overleftarrow{h}^l_{t} &=\mathcal{H}\left(W_{x \overleftarrow{h}} {h}^{l-1}_{t}+W_{\overleftarrow{h} \overleftarrow{h}} \overleftarrow{h}^l_{t+1}+b_{\overleftarrow{h}}\right) \\
h^l_{t} &=W_{\vec{h} y} \vec{h}_{t}+W_{\overleftarrow{h}_{y}} \overleftarrow{h}_{t}+b_{o}
\end{aligned}
$$
有两种方式可以将BatchNorm应用到BRNN操作中。

一种是在每次非线性操作前加上一个BatchNorm的转换：
$$
h_{t}^{l}=f\left(B\left(W^{l} h_{t}^{l-1}+U^{l} h_{t-1}^{l}\right)\right)
$$
但这样的话，均值和方差只是在一个时间步的小批次上是累加的，这没有效果。

另外一种方法是按序列归一化（sequence-wise normalization），仅对垂直连接进行批量归一化。这样对于每个隐藏单元都计算一个mini batch中所有项的均值和方差。
$$
h_{t}^{l}=f(B(W^{l} h_{t}^{l-1})+U^{l} h_{t-1}^{l})
$$
下图显示了各方法收敛情况，其中BN带来的性能改经随网络深度的增加而增加。

![img](https://img2020.cnblogs.com/blog/802043/202008/802043-20200824153651351-1913152456.png)

#### 6.3 SortaGrad

其他研究发现[6, 70]，按难度顺序展示数据可加快学习，在语音识别在内的许多序列学习问题中，较长的例子往往更具有挑战性。

使用CTC成本函数来显示话语长度：

![image.png](https://upload-images.jianshu.io/upload_images/24435917-496eda16819bcdd7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

内项是序列时间步长的乘积，随着序列的长度增加，p将会减小，称之为SortaGrad. SortaGrad使用话语的长度作为判断难度的启发式方法，因为长话语比短话语具有更高的成本。

![image-20210301153947786.png](https://upload-images.jianshu.io/upload_images/24435917-0fc678efe3cab09e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

表2:有和没有SortaGrad，有和没有批规范化的培训和开发集上的WER比较(与DS1)。

#### 6.4 Comparison of simple RNNs and GRUs

目前我们使用的是简单的双向RNN模型，用了ReLU激活方法。不过一些更复杂的RNN变种，如LSTM，GRU在一些类似任务上也是效果很好。

使用GRU来实验（因为在小数据集上，同样参数量的GRU和LSTM可以达到差不多的准确率，但GRU可以训练地更快且不易发散）

![image.png](https://upload-images.jianshu.io/upload_images/24435917-f110c59bf34db0e9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

和LSTM类似的，引入更新门 z 和重置门 r，$\sigma$ 是sigmoid函数，$f(·)$ 为双曲正切函数tanh。

表1的最后两列显示，对于固定数量的参数，GRU体系结构可在所有网络深度上实现更低的WER。

![image-20210301154025603.png](https://upload-images.jianshu.io/upload_images/24435917-d39ee6d56f4d67e4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 6.5 Frequency Convolutions & Striding

时序卷积是语音识别中常用的一种方法，它可以有效地建模变长语音的时序平移不变性。

与大型全连接网络相比，时序卷积尝试更简洁地建模由于说话人引起的频谱变化。

本文尝试添加一到三层卷积。这些都在时频域（2D）和仅时间域（1D）中。在所有情况下，我们都使用“相同”卷积。

下表展示了两个数据集的结果：一个包括2048条语音的开发集（“ Regular Dev”）和一个噪声更大的2048条语音的数据集（“ Noisy Dev”），这些数据是从CHiME 2015开发集中随机采样的。

可以看出多层的一维卷积效果一般。2D卷积可显着改善嘈杂数据的结果，而对纯净数据却没有多少提升。

![image-20210301154157176.png](https://upload-images.jianshu.io/upload_images/24435917-2ca324c7ac2a373e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在某些情况下，跨更长的步幅和更宽广的上下文，从而减小输出的大小。

在普通话模式中，采用增加步长。然而，在英语中，跨步操作可能会降低准确，因为英语语音中每个时间步长的字符数相关，在跨步时就会产生问题，考虑这点，引入语言模型来丰富英语字母表。

下表展示了在不同级别的跨步操作中，使用或不使用语言模型的bigrams下不同的时间步长。可看出 bigram允许更大的步伐，而不会牺牲单词错误率，从而有利于计算和内存的使用。

![image-20210304010508166.png](https://upload-images.jianshu.io/upload_images/24435917-5fff244a0857fd18.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 6.6 Row Convolution and Unidirectional Models

双向RNN模型很难部署到实时、低延迟的环境中，因为当用户的声音到来时，它无法实现流式处理。
仅具有前向循环的模型通常比同类的双向模型性能差，这意味着一定数量的下文对于效果提升至关重要。

一种可能的解决方案是延迟系统产生预测结果的时间，直到它具有更多上下文为止。

为了构建一个没有损失精度的单向模型，使用一种特殊的网络层，叫做Row Convolution。

![img](https://img2020.cnblogs.com/blog/802043/202008/802043-20200824165314778-1593137486.png)

![img](https://img-blog.csdn.net/20170904171106299)

根据图3和公式，这层通过线性的结合每个神经元第t时间步受未来激励，使得可以控制未来所需上下文的数量（即r）。这样可以流化所有在row convolution下面计算，以一个比较好的间隔（granularity）。

#### 6.7 系统优化

对数以千万计的网络参数，开发了一个基于高性能计算设施的高度优化的训练系统。即使现在训练深度网络都是并行的。

1. 使用同步SGD，让模型得以在多个GPU上同步训练。
2. CTC loss function的GPU实现：由于计算CTC loss function 比前后向传播更为复杂，本文实现了一个GPU版本的CTC loss function，减少了10%~20%的训练时间。
3. 使用了定制的内存分配器（custom memory allocators，自定义内存分配例程来最大程度地提高性能。

### 7 论文的解决方案有完备的理论证明吗

没有，主要是整合已有的技术构建模型，但对每一项技术都做了性能对比试验。

### 8 实验设计

#### 8.1用到了哪些数据集

ldc语料库中的华尔街日报(WSJ)语料 + LibriSpeech语料库 + 内部百度语料库。

英语模型使用11940个小时的标记语音，包含800万条语音，而普通话模型，使用了9400个小时的标记语音，包含1100万条。

##### 8.1.1 数据构造

为了过滤掉错误的文稿转录片段，利用CTC损失（the raw CTC cost），序列长度到文稿长度的比率来判断对齐程度。

数据增强：给数据集加了一些噪声，SNR在0dB到30dB之间，以增加数据，并提升对带噪语音的鲁棒性。

![img](https://img-blog.csdn.net/20170904171557694)

#### 8.2与什么算法进行了比较

相对于人工语音识别和DS1所做的算法改进对比。

模型有9层，2层2D卷积，7个循环层，有68M参数。样本都是在完整数据集里随机采样。对于每个数据集，模型进行多达20轮epoch的训练，并根据在开发集留出部分（验证集）上的错误率使用早停机制，以防止过拟合学习率取[10^-4 , 6*10^-4]，以达到最快的收敛速度，并在每个epoch之后以常数因子1.2进行退火。所有模型都使用0.99的动量。

#### 8.3评价指标是什么

1. DS1和DS2在同一示例数据集内结果比较

   ![image-20210301161210817.png](https://upload-images.jianshu.io/upload_images/24435917-c0261c7ce7c36c9a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2. 识别演讲： DS2系统在4个测试集中有3个优于人类,并且在第4个测试集中具有竞争力。

   ![image-20210301161242072.png](https://upload-images.jianshu.io/upload_images/24435917-57cc41800bb47c0e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

   两种语音系统的WER比较和人级读语音性能。

3. 带口音的识别：

   ![image-20210301161442358.png](https://upload-images.jianshu.io/upload_images/24435917-074fbaf55eaf0ef3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

4. 嘈杂环境下的演讲

   ![image-20210301161451013.png](https://upload-images.jianshu.io/upload_images/24435917-540f1e5e4abb0fa5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

   "CHiME eval dlean"是一个无噪音的baseline。 "CHiME eval real"数据集是在真实的噪声环境中采集的,而"CHiME eval sim"数据集则是将相似的噪声综合添加到干净的语音中。

   对比3、4，虽然DS2对比DS1有了改进，但在噪声环境下仍然明显低于人类水平。

5. 普通话：

   ![image-20210301161616441.png](https://upload-images.jianshu.io/upload_images/24435917-ccd1670944c7c013.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

   最好的普通话语音识别系统相比于人工，可以更好地转录简短的语音查询。

#### 8.4 实际应用

本文在服务器上进行了实际测试，希望能保持低延迟的转录。建立了一个批量调度器（batching scheduler），来集合用户请求的流数据，因此可以增加batch的大小，最终能提高效率，但同时增加了延迟。

![img](https://img-blog.csdn.net/20170904172028524)

同时部署系统使用半精度计算评估RNN，这对精度没有影响，但可以显着提高效率。

使用beam search需要在n-gram语言模型中进行重复查找，CTC查找时，只考虑累计概率至少为p的少量字符，并限制只搜索40个字符。

这样，总体的普通话语言模型查找时间快了150倍，并且对CER的影响可以忽略不计。


### 9 实验支撑

#### 9.1 论文的数据集哪里获取

[ldc语料库](https://www.ldc.upenn.edu/members/benefits)中的华尔街日报(WSJ)语料

[LibriSpeech语料库](https://www.openslr.org/12)

其他数据集为内部百度语料库。

#### 9.2 源代码哪里可以获取

https://github.com/PaddlePaddle/DeepSpeech

#### 9.3 关键代码的讲解



### 10 实验结果是否验证了科学假设？

是，证明了端到端的深度学习技术在对于多种场景下的价值（不同语言，噪声环境下）。

### 11 论文最大的贡献

1. 模型结构

2. 大量标记的训练集
3. 语音识别的计算规模

### 12 论文的不足之处

在真实嘈杂环境的数据上，DS2语音识别和人类水平依然存在差距。

#### 12.1 这篇论文之后的工作有哪些其他的改进

对于具体的应用场景，对转录可能有特殊的要求，这些需要做后处理（不直接输出字符，例如数字格式规范化）端到端的深度学习方法仍有待进一步研究，尤其是对于其适应性及基于应用场景的研究。

#### 12.2你觉得可以对这篇论文有什么改进



### 13 重要的相关论文

- Deep Speech 1 ：A. Hannun, C. Case, J. Casper, B. Catanzaro, G. Diamos, E. Elsen, R. Prenger, S. Satheesh, S. Sengupta,
  A. Coates, and A. Y. Ng. Deep speech: Scaling up end-to-end speech recognition. 1412.5567, 2014.
  http://arxiv.org/abs/1412.5567.

- Connectionist Temporal Classification (CTC)：A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber. Connectionist temporal classification: Labelling
  unsegmented sequence data with recurrent neural networks. In ICML, pages 369–376. ACM, 2006.

- 双向循环网络 - BRNNs：A. Graves, A.-r. Mohamed, and G. Hinton. Speech recognition with deep recurrent neural networks. In
  ICASSP, 2013.

- 模型并行（高性能计算）：C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich.
  Going deeper with convolutions. 2014.

- 数据并行（高性能计算）：A. Hannun, C. Case, J. Casper, B. Catanzaro, G. Diamos, E. Elsen, R. Prenger, S. Satheesh, S. Sengupta,
  A. Coates, and A. Y. Ng. Deep speech: Scaling up end-to-end speech recognition. 1412.5567, 2014.
  http://arxiv.org/abs/1412.5567.

- 门控递归单元 - GRU：K. Cho, B. Van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio. Learning
  phrase representations using rnn encoder-decoder for statistical machine translation. In EMNLP, 2014.

- 时序卷积（Temporal convolution）：A. Waibel, T. Hanazawa, G. Hinton, K. Shikano, and K. Lang. Phoneme recognition using time-delay
  neural networks,â˘AI acoustics speech and signal processing. IEEE Transactions on Acoustics, Speech and Signal Processing, 37(3):328–339, 1989.

- LibriSpeech数据集：V. Panayotov, G. Chen, D. Povey, and S. Khudanpur. Librispeech: an asr corpus based on public domain
  audio books. In ICASSP, 2015.

- LDC语料库：C. Cieri, D. Miller, and K. Walker. The Fisher corpus: a resource for the next generations of speech-totext. In LREC, volume 4, pages 69–71, 2004.

### 14 不懂之处

部署应用的一些名词，以及代码部分
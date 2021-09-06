## 26 Speech Recognition with Deep Recurrent Neural Networks

本论文利用合适的端到端（end2end)的训练方式对LSTM RNN进行训练，构建了一个深度LSTM网络，最后实现了对TIMIT数据库的高识别率

### 1  论文的结构(简要概括)

#### 1.1  Abstract

- 作者想解决什么问题？

  RNN与LSTM的结合能够实现对文本（手写草书）的识别，作者想构建深度的RNN来实现RNN对语音的识别

- 作者通过什么理论/模型来解决这个问题？

  构建深度的RNN，选择合适的end2end的训练方式，训练出深度的具有LSTM的RNN模型，用以解决上述问题

- 作者给出的答案是什么？

  最后所构建的模型实现对TIMIT数据集的高识别率，错误率为17.7%，为当时识别率最高的模型

#### 1.2 Introduction

- 作者为什么研究这个课题？

  神经网络在语言识别区域有着悠久历史，且常与隐马尔科夫模型（HMM）相结合，但HMM-RNN系统的表现并不如深度网络，而当时的具有LSTM结构的RNN和端到端的训练方式相结合，能够实现对手写草书的识别，但还未用于对语音识别的研究

- 作者使用的理论是基于哪些假设？

  深度网络在语音识别方面被证明是优于HMM-RNN系统，且RNN本身就具有深度（每一层的隐藏状态为先前隐藏状态的函数），故作者假设深度更大的具有LSTM的RNN也能够实现更好的语音识别

  神经网络在语音识别的研究通常结合马尔可夫模型（前馈玩咯吸引公众注意力）语音的特殊性决定了RNN是常可变的模型     

#### 1.3 Related work

- 和作者这篇论文相关的工作有哪些？

  ##### 循环神经网络（RNN）

  对于标准的RNN，给定输入序列$x(x 1, \ldots, x T)$，RNN会通过以下方程的迭代计算隐藏层的序列$h(h 1, \ldots, h T)$与输出序列$y(y 1, \ldots, y T)$：

  $h_{t}=\mathcal{H}\left(W_{x h} x_{t}+W_{h h} h_{t-1}+b_{h}\right)$
  $y_{t}=W_{h y} h_{t}+b_{y}$

  其中，W表示权重矩阵，b为偏置，H为隐藏层函数（通常为sigmoid函数）

  对于LSTM网络：

  $\begin{aligned} i_{t} &=\sigma\left(W_{x i} x_{t}+W_{h i} h_{t-1}+W_{c i} c_{t-1}+b_{i}\right) \\ f_{t} &=\sigma\left(W_{x f} x_{t}+W_{h f} h_{t-1}+W_{c f} c_{t-1}+b_{f}\right) \\ c_{t} &=f_{t} c_{t-1}+i_{t} \tanh \left(W_{x c} x_{t}+W_{h c} h_{t-1}+b_{c}\right) \\ o_{t} &=\sigma\left(W_{x o} x_{t}+W_{h o} h_{t-1}+W_{c o} c_{t}+b_{o}\right) \\ h_{t} &=o_{t} \tanh \left(c_{t}\right) \end{aligned}$

  $\sigma$为sigmoid函数，i为输入门，f为遗忘门，o为输出门，c为激活向量（记忆细胞）。

  对于BRNN：通过迭代后向层t=T到1，前向层来自t=1至T计算前向隐藏序列H ，后向隐藏序列H和输出序列 y  ，然后以此更新输出层

  $\begin{aligned} \vec{h}_{t} &=\mathcal{H}\left(W_{x \vec{h}} x_{t}+W_{\vec{h}\vec{h}}  \vec{h}_{t-1}+b_{\vec{h}}\right) \\ \overleftarrow{h}_{t} &=\mathcal{H}\left(W_{x \overleftarrow{h}} x_{t}+W_{\overleftarrow{h} \overleftarrow{h}} \overleftarrow{h}_{t+1}+b \overleftarrow{\hbar}\right) \\ y_{t} &=W_{\vec{h} y} \vec{h}_{t}+W_{\overleftarrow{h} y} \overleftarrow{h}_{t}+b_{y} \end{aligned}$

  ![image-20210202092037339](C:\Users\fan\Documents\WeChat Files\wxid_h21n3s7m58ib22\FileStorage\File\2021-02\论文笔记模板md.assets\image-20210202092037339.png)

  双向LSTM：将BRNN与LSTM组合在一起可以得到双向LSTM 

  深度RNN:将多个RNN隐藏层堆叠在彼此之上来创建深度 RNN，其中一个层的输出序列形成下一个的输入序列

  其中隐藏层函数由以下公式迭代计算(n=1到N且t=1到T)

  $h_{t}^{n}=\mathcal{H}\left(W_{h^{n-1} h^{n}} h_{t}^{n-1}+W_{h^{n} h^{n}} h_{t-1}^{n}+b_{h}^{n}\right)$

  设$h^{0}=x$，则输出为：$y_{t}=W_{h^{N} y} h_{t}^{N}+b_{y}$

  以此可构建出深度BRNN与深度LSTM(本论文主要使用结构)

  ##### 网络训练

  补充algorithm

  ###### 对齐（alignment）【CTC和RNN-T不同】

  实际应用时，输入X经过encoder后转成的向量T的长度往往会大于输出Y的长度，则需要一个补齐的操作，一个输入向量经过一个classifier后会转成一个token

  ![image-20210203175419649](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210203175419649.png)

  ###### forward-backward algorithm：

  ![image-20210205162246359](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210205162246359.png)

  ![image-20210204091317764](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210204091317764.png)

  ![image-20210204092039540](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210204092039540.png)

  ![image-20210204094119919](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210204094119919.png)

  则最后要算的P（h|x）就是将每个token认为的几率乘起来

  如何将所有的alignment加起来（forward-backward algorithm）

  ###### ![image-20210203223245105](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210203223245105.png)

  ![image-20210205162550099](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210205162550099.png)

  ![image-20210205162640435](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210205162640435.png)

  ![image-20210205162950085](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210205162950085.png)

  ![image-20210205163224520](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210205163224520.png)

  ![image-20210205163408961](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210205163408961.png)

  ![image-20210205163537105](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210205163537105.png)

  ![image-20210205163627367](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210205163627367.png)

  hstar即为每个distribution中最大的连乘

  ###### 1.CTC（联结时序分类模型）

  不需要先进行输入序列和输出序列的对齐，使用softmax层来定义单独的输出分布Pr(k|t)，这个分布包含了ķ个音素加上一个额外的空集 ∅ 表示非输出(因此softmax层是大小 ķ+ 1)，可以**duplicate**使用前向-后向算法**对所有可能的对齐**（alignment）进行求和并确定概率Pr(z|x)

  本文使用双向网络：

  $\begin{aligned} y_{t} &=W_{\vec{h}^{N_{y}}} \vec{h}_{t}^{N}+W_{\overleftarrow{h}^{N} y} \overleftarrow{h}_{t}^{N}+b_{y} \\ \operatorname{Pr}(k \mid t) &=\frac{\exp \left(y_{t}[k]\right)}{\sum_{k^{\prime}=1}^{K} \exp \left(y_{t}\left[k^{\prime}\right]\right)} \end{aligned}$

  ###### 2.RNN Transducer

  可视为对CTC网络的一种改进，将类似于CTC的网络与单独的RNN相结合，**使得每一个timestep的输出y不是独立的**可同时训练声学和语言模型，该模型通过前一个音素来预测后一个音素，softmax层的大小也为k+1（包含空集），可由随机初始化权值进行训练，**不**可以**duplicate**使用前向-后向算法**对所有可能的对齐**（alignment）进行求和并确定概率Pr(z|x)

  ![image-20210202210722227](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210202210722227.png)

  $\begin{aligned} l_{t} &=W_{\vec{h}^{N} l} \vec{h}_{t}^{N}+W_{\overleftarrow{h} N_{l}} \overleftarrow{h_{t}^{N}}+b_{l} \\ h_{t, u} &=\tanh \left(W_{l h} l_{t, u}+W_{p b} p_{u}+b_{h}\right) \\ y_{t, u} &=W_{h y} h_{t, u}+b_{y} \\ \operatorname{Pr}(k \mid t, u) &=\frac{\exp \left(y_{t, u}[k]\right)}{\sum_{k^{\prime}=1}^{K} \exp \left(y_{t, u}\left[k^{\prime}\right]\right)} \end{aligned}$

  

- 之前工作的优缺点是什么？

  多为使用HMM来实现语音识别

- 作者主要是对之前的哪个工作进行改进？

  鉴于之前RNN对手写草书的良好识别效果与深度网络对语音识别的良好效果，考虑加深RNN深度，探究其深度RNN对语音识别的效果

#### 1.5 Experiment

- 作者是在哪些数据集或者说场景下进行了测试？

  TIMIT数据集

- 实验中的重要指标有哪些？

  隐藏层的层数、LSTM单元个数、识别率

- 文章提出的方法在哪些指标上表现好？在哪些指标上表现不好？

  ![image-20210202102650835](C:\Users\fan\Documents\WeChat Files\wxid_h21n3s7m58ib22\FileStorage\File\2021-02\论文笔记模板md.assets\image-20210202102650835.png)

  

#### 1.6 Conclusion

- 这篇论文最大的贡献是什么？

  通过实验证明了深度RNN对语音识别的效果良好，到达了当时对TIMIT数据集最高的识别率

- 论文中的方法还存在什么问题？

  缺乏对更大、不同的数据集进行实验、应用

- 作者觉得还可以怎么改进？

  下一步可尝试系统扩展到大词汇量语音识别，或将频域卷积神经网络与深度LSTM相结合。

### 2 论文想要解决的问题？

#### 2.1 背景是什么？

当时的具有LSTM结构的RNN和端到端的训练方式相结合，能够实现对手写草书的识别，但还未用于对语音识别的研究

### 3 论文研究的是否是一个新问题

在当时看来，是一个新问题

### 4 论文试图验证的科学假设

深度的具备LSTM结构的RNN能否实现语音识别

### 5 相关的关键人物与工作

#### 5.1 之前存在哪些相关的工作

RNN网络的构建、CTC模型、HMM

#### 5.2 本文是对哪个工作进行的改进

受深度网络的启发，考虑加深RNN的深度，用于语音识别

#### 5.3 这个领域的关键研究者

提出了CTC、RNN-T,完成了手写字体的识别，《上一篇论文》构建了深度的RNN

![image-20210202102922108](C:\Users\fan\Documents\WeChat Files\wxid_h21n3s7m58ib22\FileStorage\File\2021-02\论文笔记模板md.assets\image-20210202102922108.png)

![image-20210203121435088](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210203121435088.png)

### 6 论文提出的解决方案的关键

加深RNN深度

### 7 论文的解决方案有完备的理论证明吗

没有，更多的是实验证明

### 8 实验设计

#### 8.1用到了哪些数据集

TIMIT数据集

#### 8.2与什么算法进行了比较

与隐藏层层数不同的网络、LSTM节点数不同的网络、使用tanh函数的网络进行了比较

#### 8.3评价指标是什么

对TIMIT数据集的识别错误率

### 9 实验支撑

#### 9.1 论文的数据集哪里获取

http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3

#### 9.2 源代码哪里可以获取

https://github.com/mravanelli/pytorch-kaldi/blob/master/neural_networks.py

$\arg \max _{\mathrm{Y}} \max _{h \in \operatorname{align}(Y)} \log P(h \mid X)$

![image-20210202214709660](C:\Users\fan\Desktop\26 Speech Recognition with Deep Recurrent Neural Networks.assets\image-20210202214709660.png)

### 10 实验结果是否验证了科学假设？

是，验证了

### 11 论文最大的贡献

提出了RNN-T模型，说明了深度的具有LSTM的RNN网络可以很好的实现语音的识别

### 12 论文的不足之处

这篇论文提出的系统暂时没有用于对更大的数据集的测试，且仅对TIMIT数据集的测试，错误率也高达17.7%，仍可通过调整其它参数尝试（继续加深隐藏层深度）来提高其准确率

### 13 重要的相关论文

《Fast and Accurate Recurrent Neural Network Acoustic Models for Speech》

### 14 不懂之处

没有特别清楚是怎么通过RNN transducer实现深层的RNN对语音的识别，感觉好像只是通过实验证明了深层的RNN具有良好的效果。

实验刚开始的设置部分不是特别理解
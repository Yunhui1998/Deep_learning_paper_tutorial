## 27 Towards End-to-End Speech Recognition with Recurrent Neural Networks

基于循环神经网络的端到端语音识别研究

### 1  论文的结构

#### 1.1  Abstract

- 作者想解决什么问题？

  利用端到端的模型来构建语音识别系统，实现直接用文本对音频数据进行转录，并且能降低语音识别的错误率。

- 作者通过什么理论/模型来解决这个问题？

  基于深度双向LSTM的循环神经网络结构和 CTC 相结合。并提出损失函数来对网络进行再优化。


#### 1.2 Introduction

- 作者为什么研究这个课题？

  算法和计算机的发展使得神经网络能够以端到端的方式来完成以前需要大量人类专门知识的任务，可以应用在自动语音识别技术上。

- 和作者这篇论文相关的工作有哪些？

  传统的语音识别技术，即 DNN-HMM 类的模型。

- 之前工作的优缺点是什么？

  传统非端到端的语音识别技术：

  1. 不一致性：输入特征提取，训练神经网络对声音数据单帧进行分类，将其输出分布重新表述为隐马尔可夫模型(HMM)的发射概率。因此，用于训练网络的目标函数与真正的性能度量(序列级转录准确性)有不一致性。
  2. 提取语音信号的帧精度提高不太能导致最后语音识别准确度的提高。
  3. 需要先实现 HMM 结构与语音的对齐，然后才能进一步地训练深度神经网络。
  4. 需要字典，从语音转录为词汇时，需要一个发音字典来从单词映射到音素序列。创建这样的字典需要大量的人力，在解码的时候也同样需要依赖这个发音词典。
  5. 语音中通常包含大量先验信息，而将语言文字和声音分开建模会影响性能。

- 作者主要是对之前的哪个工作进行改进？

  端到端的RNN模型优点：

  一步直接实现原始语音的输入与解码识别，无需对齐和字典，真正的做到数据拿来可用。

  训练速度提高，准确率提高。
  
  更换识别语言体系时可以利用相同的框架结构直接训练。（例如同样的网络结构可以训练包含 26 个字符的英文模型，也可以训练包含 3000 个常用汉字的中文模型，甚至可以将中英文的词典合在一起，训练混合模型。）


#### 1.4  Theoretical Analysis

- 作者是用什么理论证明了自己的方法在理论上也是有保障的？

  提出再优化目标函数的方法时，对计算预期转录损失L(x)的最小值，有理论推导。

#### 1.5 Experiment

- 作者是在哪些数据集或者说场景下进行了测试？

  将以往的DNN-HMM 类语音识别模型作为baseline，进行对比测试。

- 实验中的重要指标有哪些？

  语音识别评价指标：单词错误率(word error rate, WER)
  
  为了使识别出来的词序列和标准的词序列之间保持一致，需要进行替换、删除或者插入某些词，这些插入、替换或删除的词的总个数，除以标准的词序列中词的总个数的百分比，即为WER：
  ![这里写图片描述](https://pic.36krcnd.com/201711/24035632/i1dxncvro9p1bau0!1200)

#### 1.6 Conclusion

- 这篇论文最大的贡献是什么？

  证明了字符级语音转录可以通过一个端到端的循环神经网络执行最少的预处理。介绍了新的目标函数，允许对网络进行直接的单词错误率再优化，并展示了在解码期间如何将网络输出与语言模型集成。

- 论文中的方法还存在什么问题？

  实现音频识别后，网络对单词的拼写能力仍然不足。

- 作者觉得还可以怎么改进？

  实验结果显示语言模型和神经网络相结合效果很好，可以进一步探究。

### 2 论文想要解决的问题？

#### 2.1 背景是什么？

传统的语音识别需要先实现 HMM 结构与语音的对齐，利用多个模型，多个分解步骤进行网络训练。除此之外，在训练这一类的模型时，需要对语音进一步拆解成为音素，利用发音词典来从单词映射到音素序列，消耗人力。在解码的时同样需要依赖这个发音词典。

#### 2.2 之前的方法存在哪些问题

端到端的模型由于不引入传统的音素或词的概念，直接训练音频到文本的模型，可以有效地规避上述难点。

作者想办法提升端到端模型的性能。

#### 2.3 输入和输出是什么？

输入原始音频，输出识别出的文本。

### 3 论文研究的是否是一个新问题

不是，端到端的技术在之前就有提出，作者希望提升其性能。

### 4 论文试图验证的科学假设

构建端到端的语音识别系统，能够直接用文本对音频数据进行转录，并且能降低语音识别的错误率。

### 5 相关的关键人物与工作

#### 5.1 之前存在哪些相关的工作

DNN-HMM 类传统的非端到端语音识别技术。

#### 5.2 本文是对哪个工作进行的改进

- 端到端的RNN模型：一步直接实现原始语音的输入与解码识别，无需对齐和字典，真正的做到数据拿来可用。提高训练速度和准确率，更换识别语言体系时可以利用相同的框架结构直接训练。

- 实现：深度双向LSTM网络，可以利用未来的信息。

- CTC：集成所有可能的输入输出对齐，简化了对齐工作。
- 建立预期转录损失方程，实现再优化，提高精度。

#### 5.3 这个领域的关键研究者

![image.png](https://upload-images.jianshu.io/upload_images/24435792-978ee7bec8d42a54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

1 - 强化学习，DQN算法发明人和RNN / drnn在语音识别

2 - sequence learning和LSTM研究。

### 6 论文提出的解决方案的关键

#### 6.1 LSTM模型（Long Short-Term Memory）

标准的循环神经网络（RNN）计算隐藏层序列 $h$ 的迭代式为：

![img](https://img-blog.csdnimg.cn/2019030415333315.png)

$W$项表示权重矩阵(如$ W_{xh}$为隐藏层权重矩阵)，$b$项表示偏置向量，(如 $b_h$ 为隐藏层偏置向量)，$H$ 为隐藏层函数。

$H$通常是sigmoid函数的基本应用。Long Short-Term Memory (LSTM)架构使用专门构建的内存单元memory cells来存储信息，它更善于发现和利用远程上下文。图1显示了一个LSTM内存单元。

![img](https://img-blog.csdnimg.cn/20190304183240636.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

LSTM的函数$H$由以下复合函数实现：

![img](https://img-blog.csdnimg.cn/20190304153632975.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

其中$σ$是sigmoid函数, $i ,f, o $ 和 $c$ 分别为input gate, forget gate, output gate, cell activation vectors（输入门、遗忘门、输出门和细胞激活向量），大小与隐藏层向量 $h$ 相同。权矩阵下标具有明显的含义，如$W_{hi}$隐输入门矩阵。$W_{xo}$是输入输出门矩阵等。

#### 6.2 双向RNNS：BRNNs (Bidirectional RNNs)

传统神经网络的一个缺点是它们只能利用以前的信息。在语音识别中，所有的话语同时被转录，没有理由不利用未来的信息。

双向RNNs (BRNNs) (Schuster & Paliwal, 1997)通过使用两个独立的隐藏层来处理两个方向的数据，然后将数据转发到相同的输出层。

![img](https://img-blog.csdnimg.cn/20190304183252881.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

如图所示，BRNN通过迭代后向层从t = T到 1，前向层从t = 1到 T，然后更新输出层来计算前向隐藏序列 $\vec{h}$，后向隐藏序列 $\overleftarrow{h}$ 和输出序列：

![img](https://img-blog.csdnimg.cn/20190304183224470.png)

#### 6.3 深度双向LSTM

将BRNNs与LSTM相结合，得到了深度双向LSTM 网络(Graves & Schmidhuber, 2005)，它可以在两个输入方向上访问远程上下文。

混合系统最近成功的一个关键因素是使用了深度架构，这种架构能够逐步构建更高级别的声学数据表示。通过将多个RNN隐藏层叠加在一起，一个层的输出序列构成下一个层的输入序列，来创建深层RNN。

如图3所示。假设使用相同的隐藏层函数，对于堆栈中的所有N层，从N = 1 迭代到 N，从t = 1 迭代到 T，迭代计算隐藏向量序列 $h^n$ :

![img](https://img-blog.csdnimg.cn/20190304183434830.png)

$h ^ 0 = x $ 时，输出 $y_t$ 是：

![img](https://img-blog.csdnimg.cn/20190304183518523.png)

![image.png](https://upload-images.jianshu.io/upload_images/24435792-03879844015e2d92.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

深层双向RNNs可以通过将每个隐藏序列 $h^n$ 替换为正向序列和反向序列 $\vec{h}^n$， $\overleftarrow{h}^n$ ：并确保每个隐藏层同时接收下一层的前向和后向层的输入。如果再将LSTM用于隐藏层，则完整的架构称为深度双向LSTM (Graves et al.， 2013)。即本文使用的主要架构。

#### 6.4 时序分类模型：CTC (Connectionist Temporal Classification )

参考文献：[Sequence Modeling With CTC](https://distill.pub/2017/ctc/)

Connectionist Temporal Classification (CTC) (Graves, 2012)是一种目标函数，它允许训练RNN执行序列转录任务，而不需要事先了解输入序列和目标序列之间进行任何的比对对齐。

![img](https://img-blog.csdnimg.cn/20190304191519715.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

上图利用CTC分辨“HIS FRIEND'S”，可以看出CTC方法不需要对每个音素进行标记，而是根据输入音频，预测了一系列 峰值（spikes） 和一些 可能空白 （blanks）用来区分字母。 

提出损失函数，对于给定的输入，我们想训练模型以最大化其分配给正确答案的概率。

$$
p(\pi \mid \mathbf{x})=\prod_{t=1}^{T} y_{\pi_{t}}^{t}, \forall \pi \in L^{\prime T}
$$

这个公式表示给定一个输入 $x$ ，L是序列标签的label，从时间点 t=1 到最后 t=T 每个时间点的概率相乘，得到对应的路径概率（比如上图的P(H), P(I)）。

接下来计算多对一映射的概率，把所有多对一的映射的集合，算出来对应一个真正的sequence label (L) 的概率。
$$
p(\mathbf{l} \mid \mathbf{x})=\sum_{\pi \in \mathcal{B}^{-1}(\mathbf{1})} p(\pi \mid \mathbf{x})
$$
接下来生成分类器，解码一个CTC网络(即为给定输入序列 x 找到最可能的输出转录 y，可以通过在每个时间步上选取单个最可能的输出，找到概率最大的组合：
$$
\operatorname{argmax}_{y} \operatorname{Pr}(\boldsymbol{y} \mid \boldsymbol{x}) \approx \mathcal{B}\left(\operatorname{argmax}_{\boldsymbol{a}} \operatorname{Pr}(\boldsymbol{a} \mid \boldsymbol{x})\right)
$$
利用集束搜索法：

![img](https://pic4.zhimg.com/80/v2-47c168379370d140fd1eeaaa93e72fa9_1440w.jpg?source=1940ef5c)

也就是并非每步找到最大的概率，而是每个节点都是在上一个节点的总概率前提下，找到概率最大的序列，作为当前转录的组合。

#### 6.5 预期转录损失(Expected Transcription Loss)

提出再优化方法，允许训练RNN后，再优化定义在输出转录(如WER)之上的任意损失函数的期望值。

给定输入序列 $ x$ , CTC定义的转录序列y上的$Pr(y|x)$分布，以及实值转录损失函数$L(x，y）$，预期转录损失 $L(x)$ 定义为：

![img](https://img-blog.csdnimg.cn/20190304185600896.png)

作者利用蒙特卡罗抽样法近似出 $L$ 和它的梯度，将式(15) 和式（14）代入式(17)，进行推导变换后得到：

![img](https://img-blog.csdnimg.cn/20190304190342895.png)

![img](https://img-blog.csdnimg.cn/20190304190351717.png)

这个式子的最大意义在于给定序列 $a^i$ 对 $y^k_t$  的导数主要相关于$\operatorname{Pr}\left(k^{\prime}, t \mid \boldsymbol{x}\right)$。这意味着网络只接收一个错误项，用于更改更改损失的对齐方式。

也就是说只需重新计算与对齐更改相对应的那部分错误损失，就可以对其进行优化，无需从整体进行优化。本文给出的结论为建议使用期望损失最小化来重新培训已经使用CTC进行培训的网络，而不是从一开始就应用它。

### 7 论文的解决方案有完备的理论证明吗

有，深度双向LSTM和CTC都是使用之前的技术，其中寻找优化算法结构方法，对于损失函数的期望值的优化推导，有具体的过程。

### 8 实验设计

#### 8.1用到了哪些数据集

Wall Street Journal (WSJ)语料库中的LDC语料库（LDC93S6B和LDC94S13B)上进行。RNN在14小时子集“train-si84”和完整的81小时子集上进行训练，验证集为“test-dev93”开发集。

#### 8.2与什么算法进行了比较

将之前运用在语音识别的DNN-HMM网络作为baseline，和本文的RNN网络进行比较。使用CTC训练RNN，解码部分采用波束搜索，然后使预期转录损失最小化的方法对RNN进行再训练。

#### 8.3评价指标是什么

![img](https://img-blog.csdnimg.cn/20190304191441740.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

上表为测试《华尔街日报》数据集的结果。分数代表单词错误率/字符错误率(已知)。“LM”是用于解码的语言模型（monogram，bigram，trigram），。“14HR”和“81HR”是指用于培训的数据量。RNN- wer比原始的RNN- ctc网络多了优化函数再训练。

网络共有五层双向LSTM隐含层，每层500个单元，训练时使用带动量的随机梯度下降法，动量为0.9，学习率为 $10^{-4}$，每句话更新一个权值。分别实验RNN在没有字典和语言模型的情况下解码，当RNN结合语言模型时设置RNN：语言模型的权重为7.7：12，并记录最佳的实验结果。

横向对比：从音频时间长度来看，基线系统从14小时到81小时的训练集，性能略有改进，而RNN模型的错误率则大幅下降。一种可能的解释是：14个小时的转录语音不足以让RNN学会如何拼写足够的单词来进行准确的转录，但足以学会识别音素。

单看81HR列：当没有语言模型时，字符级别RNN的性能优于基线模型。这可能是由于RNN能够学习更强大的声学模型，因为它可以访问更广泛的声学上下文。

然而，基线模型在LM增强时性能超过了RNN：可能是RNN内的结构和LM语言模型互相干扰。尽管如此，考虑到如此多的先验信息(音频预处理、发音字典、状态、强制对齐)被编码到基线系统中，差异还是很小。

组合模型的性能比单独使用RNN或基线要好得多：在基线上超过1%的绝对改善远远大于模型平均通常所看到的微小改善;这可能是由于结合了两种系统之间的差异性。

### 9 实验支撑

#### 9.1 论文的数据集哪里获取

使用[ldc语料库](https://www.ldc.upenn.edu/members/benefits)中的华尔街日报(WSJ)语料[LDC93S6B](https://catalog.ldc.upenn.edu/LDC93S6B)和[ LDC94S13B](https://catalog.ldc.upenn.edu/LDC94S13B)，非会员需要收取费用。

#### 9.2 源代码哪里可以获取

#### 9.3 关键代码的讲解

出自https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0

CTC解码：

```python
"""
Author: Awni Hannun
This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.
The algorithm is a prefix beam search for a model trained
with the CTC loss function.
For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873
"""

import numpy as np
import math
import collections

NEG_INF = -float("inf")

def make_new_beam():
  fn = lambda : (NEG_INF, NEG_INF)
  return collections.defaultdict(fn)

def logsumexp(*args):
  """
  Stable log sum exp.
  """
  if all(a == NEG_INF for a in args):
      return NEG_INF
  a_max = max(args)
  lsp = math.log(sum(math.exp(a - a_max)
                      for a in args))
  return a_max + lsp

def decode(probs, beam_size=100, blank=0):
  """
  Performs inference for the given output probabilities.
  Arguments:
      probs: The output probabilities (e.g. post-softmax) for each time step. Should be an array of shape (time x output dim).
      beam_size (int): Size of the beam to use during inference.
      blank (int): Index of the CTC blank label.
  Returns the output label sequence and the corresponding negative
  log-likelihood estimated by the decoder.
  """
  T, S = probs.shape 
  probs = np.log(probs)

  # Elements in the beam are (prefix, (p_blank, p_no_blank))
  # Initialize the beam with the empty sequence, a probability of
  # 1 for ending in blank and zero for ending in non-blank
  # (in log space).
  beam = [(tuple(), (0.0, NEG_INF))]

  for t in range(T): # Loop over time

    # A default dictionary to store the next step candidates.
    next_beam = make_new_beam()

    for s in range(S): # Loop over vocab
      p = probs[t, s]

      # The variables p_b and p_nb are respectively the
      # probabilities for the prefix given that it ends in a
      # blank and does not end in a blank at this time step.
      for prefix, (p_b, p_nb) in beam: # Loop over beam

        # If we propose a blank the prefix doesn't change.
        # Only the probability of ending in blank gets updated.
        if s == blank:
          n_p_b, n_p_nb = next_beam[prefix]
          n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
          next_beam[prefix] = (n_p_b, n_p_nb)
          continue

        # Extend the prefix by the new character s and add it to
        # the beam. Only the probability of not ending in blank
        # gets updated.
        end_t = prefix[-1] if prefix else None
        n_prefix = prefix + (s,)
        n_p_b, n_p_nb = next_beam[n_prefix]
        if s != end_t:
          n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
          
        # *NB* this would be a good place to include an LM score.
        next_beam[n_prefix] = (n_p_b, n_p_nb)

        # If s is repeated at the end we also update the unchanged
        # prefix. This is the merging case.
        if s == end_t:
          n_p_b, n_p_nb = next_beam[prefix]
          n_p_nb = logsumexp(n_p_nb, p_nb + p)
          next_beam[prefix] = (n_p_b, n_p_nb)

    # Sort and trim the beam before moving on to the
    # next time-step.
    beam = sorted(next_beam.items(),
            key=lambda x : logsumexp(*x[1]),
            reverse=True)
    beam = beam[:beam_size]

  best = beam[0]
  return best[0], -logsumexp(*best[1])

if __name__ == "__main__":
  np.random.seed(3)

  time = 50
  output_dim = 20

  probs = np.random.rand(time, output_dim)
  probs = probs / np.sum(probs, axis=1, keepdims=True)

  labels, score = decode(probs)
  print("Score {:.3f}".format(score))
```

出自：https://github.com/holm-aune-bachelor2018/ctc

keras， brnn + ctc



出自：https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/blob/master/BasicLSTM/main.py

lstm代码

### 10 实验结果是否验证了科学假设？

是，证明可以通过一个端到端的模型实现字符级语音转录，模型更加简洁快速，且精度和传统的HMM模型相当。

### 11 论文最大的贡献

证明了字符级语音转录可以通过一个端到端的循环神经网络执行最少的预处理。介绍了新的目标函数，允许对网络进行直接的单词错误率再优化，并展示了在解码期间如何将网络输出与语言模型集成。

### 12 论文的不足之处

对于字符级别的转录，该网络除了学习如何识别语音外，还需要学习如何转换为字母，即实现拼写功能。

像所有的语音识别系统一样，该网络也会犯语音错误，比如把“single”(单一的)说成“shingle”，有时还会混淆同音异义词，比如“two”和“to”。后一个问题可能比通常更难用语言模型来解决，因为在发音上接近的单词在拼写上可能相当遥远。也会产生词汇错误，例如。把“boutique”（精品）写成“ bostik”，把“illustrate”写成“ alstrait”。

它能够正确地转录相当复杂的单词，如“运动”、“分析师”和“股权”，这些单词经常出现在金融文本中（可能是作为特例学习它们）。但对于发音和拼写不熟悉的单词方面都很困难，比如‘Milan’ ，‘Dukakis’ 这样的专有名称。这表明即使没有字典，词汇表外的单词仍然可能对字符级识别造成困难。

#### 12.1 这篇论文之后的工作有哪些其他的改进

探讨实验中RNN和语言模型的结合：在训练过程中将语言模型整合到CTC或预期转录损失目标函数中。

#### 12.2你觉得可以对这篇论文有什么改进



### 13 重要的相关论文

双向RNNs (BRNNs) (Schuster & Paliwal, 1997)：[Bidirectional recurrent neural networks](https://ieeexplore.ieee.org/abstract/document/650093)

本文使用的深度双向LSTM (Graves et al.， 2013)架构：[Speech recognition with deep recurrent neural networks](https://ieeexplore.ieee.org/abstract/document/6638947)

Connectionist Temporal Classification (CTC) (Graves, 2012)是一种目标函数，之后Eq.(15)可以通过动态规划算法进行有效的评估和区分(Graves et al.， 2006)。

[Connectionist Temporal Classification](https://link.springer.com/chapter/10.1007/978-3-642-24797-2_7)

CTC的工作原理：[Sequence ModelingWith CTC](https://distill.pub/2017/ctc/)

### 14 不懂之处

数学方面的知识，对预期转录损失L(x)的最小值优化的理论推导
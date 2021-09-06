## 14 Decoupled Neural Interfaces using Synthetic Gradients

DeepMind 在 2016 年8月18日发布的；在现有的神经网络模型基础上，提出了一种称为 Decoupled Neural Interfaces(后面缩写为 DNI) 的网络层之间的交互方式，用来加速神经网络的训练速度。

### 1  论文的结构

#### 1.1  Abstract

- 作者想解决什么问题？
  BP算法里，网络的所有层、模块，都是锁定的，在它们可以被更新之前，它们必须等待网络的其余部分执行正向传播和反向传播误差。

  举个例子，正向传播中只有计算了前一层的输出，下一层才能开始计算；反向传播也类似，其它层必须等待。这个等待的过程就是处于锁定中（locked）。

  作者给这些限制分了三个类：

  * Forward Locking: 正向锁定。前一层没有用输入完成计算，后一层无法进行根据输入进行计算
  * Update Locking: 更新锁定。如果一个网络层依赖的层没有完成前馈计算，该层无法进行更新
  * Backward Locking: 反向锁定。如果一个网络层依赖的层没有完成前向传播或反向传播，该层无法进行更新

  本篇论文的工作就是尝试解决这个层与层之间的限制、约束。

- 作者通过什么理论/模型来解决这个问题？

  本篇论文提出了**DNI——解耦神经接口**来解决这个问题。其中一个预测梯度的模型如下：

  <img src="https://upload-images.jianshu.io/upload_images/16793245-d0921e6f1ae8bbb5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" alt="image-20210206151602970.png" style="zoom:50%;" />

  上图中$M_B$为DNI网络，$f_A$为上一层，$f_B$为下一层，$h_A$为$f_A$的输出，$\delta_A$为$f_A$的梯度$\frac{\partial L}{\partial h_A}$，$\hat{\delta}_A$为$f_A$的预测梯度。

  该模型的可以理解为用另一个网络去预测梯度。这个网络的输入为某一层的输出，输出为该层对应的梯度。预测梯度可以直接解决上述限制的Updata Locking和Backward Locking。

- 作者给出的答案是什么？

  作者最后证明了DNI不仅可以用于预测梯度，还可以用于预测输入（只利用输入层的数据预测任意层的输入）。这样就可以解决Forward Locking。加上预测梯度，就可以拜托三个限制。

#### 1.2 Introduction

- 作者为什么研究这个课题？

  三种锁定限制我们以顺序的（逐层进行）、同步的（每层不能单独更新）方式运行和更新神经网络。虽然在训练简单的前馈网络时看起来是良性的，但当考虑创建在不同的、可能是不规则或异步时间尺度的多个环境中运行的网络系统时，就会产生问题。

- 目前这个课题的研究进行到了哪一阶段？存在哪些缺陷？作者是想通过本文解决哪个问题？

  论文中整理了其他的也是消除三类锁定或者反向传播的算法，有的消除反向传播但仍保留反向、更新锁定（target prop）；有的只能消除反向锁定而不消除更新锁定（REINFORCE、Kickback、Policy Gradient Coagent Networ）；有的是不可扩展的，要求优化器具有网络状态的全局知识（Real-Time Recurrent Learning、approximations）；Taylor、Carreira-Perpin an & Wang的工作允许在没有反向传播的情况下并行训练层，但在实践中不能扩展到更复杂和通用的网络架构。
  相较之下，本篇论文通过将各层之间的相互作用定义为DNI的局部通信问题，以此消除了学习系统对全局知识的需要。

#### 1.3 Related work

上文已基本基本阐述。

- 和作者这篇论文相关的工作有哪些？

  最接近作者的工作的是“用于梯度上升的值函数”和“训练神经网络的critics”。

- 作者主要是对之前的哪个工作进行改进？

  之前的工作大多数都无法完全消除三种锁定。

#### 1.4  Theoretical Analysis

- 作者是用什么理论证明了自己的方法在理论上也是有保障的？

  要解除更新锁定，首先从模拟更新开始。而权重的更新公式最重要的是梯度部分：
  $$
  \begin{aligned}
  \frac{\partial L}{\partial \theta_{i}} &=f_{\text {Bprop }}\left(\left(h_{i}, x_{i}, y_{i}, \theta_{i}\right), \ldots\right) \frac{\partial h_{i}}{\partial \theta_{i}} \\
  & \simeq \hat{f}_{\text {Bprop }}\left(h_{i}\right) \frac{\partial h_{i}}{\partial \theta_{i}}
  \end{aligned}
  $$
  $L$是损失函数，$\theta_i$是第$i$层的权重参数，$f_{\text {Bprop }}\left(\left(h_{i}, x_{i}, y_{i}, \theta_{i}\right), \ldots\right)$是损失函数对第$i$层的输出的梯度$\frac{\partial L}{\partial h_i}$；而$f_{\text {Bprop }}\left(\left(h_{i}, x_{i}, y_{i}, \theta_{i}\right), \ldots\right)$可以用$\hat{f}_{\text {Bprop }}\left(h_{i}\right)$逼近、模拟。这样梯度的计算就只依赖于本地的信息$h_i$。

  注：作者将前一层视为信息的发送者（也视为一个model），后一层视为信息的接受者（也视为model）。

  该方法的前提是基于一种简单的学习通信协议（两个model之间的通信），允许神经网络模块在不更新锁定的情况下进行交互和训练。一般通信协议是对生成训练信号的方式，在这里，我们专注于一个特殊的网络实现梯度下降训练——用解耦神经接口(DNI)更换标准的神经接口(两个模块之间的连接神经网络)。$\hat{f}_{\text {Bprop }}\left(h_{i}\right)$就是合成梯度。

  **递归网络**：

  在RNN的应用，则略有不同。这是因为RNN可以拆解为无限层的网络。RNN部分的推导比较复杂（没有看懂），就不展开讲了。
  $$
  \begin{aligned}
  \theta-\alpha \sum_{\tau=t}^{\infty} \frac{\partial L_{\tau}}{\partial \theta} &=\theta-\alpha\left(\sum_{\tau=t}^{t+T} \frac{\partial L_{\tau}}{\partial \theta}+\left(\sum_{\tau=T+1}^{\infty} \frac{\partial L_{\tau}}{\partial h_{T}}\right) \frac{\partial h_{T}}{\partial \theta}\right) \\
  &=\theta-\alpha\left(\sum_{\tau=t}^{t+T} \frac{\partial L_{\tau}}{\partial \theta}+\delta_{T} \frac{\partial h_{T}}{\partial \theta}\right)
  \end{aligned}
  $$
  **前馈网络**：

  前馈网络（a）中，$i$层在更新时需要$\mathcal{F}^N_{i+1}$的结果，因此是更新锁定的。

  ![image-20210209155845461.png](https://upload-images.jianshu.io/upload_images/16793245-080743317c461ae8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

  为了移除更新锁定，更新公式改为：
  $$
  \theta_{n} \leftarrow \theta_{n}-\alpha \hat{\delta}_{i} \frac{\partial h_{i}}{\partial \theta_{n}}, n \in\{1, \ldots, i\}
  $$
  对应（b）和（c）。为了训练合成梯度模型$M_{i+1}$的参数，损失函数定义为$\left\|\hat{\delta}_{i}-\delta_{i}\right\|_{2}^{2}$。但是这样又回到了原点：还是需要上一层计算了真实梯度$\delta_i$才可以训练，有点套娃。所以作者再次做了一个近似：
  $$
  \delta_{i}=\hat{\delta}_{i+1} \frac{\partial h_{i+1}}{\partial h_{i}}
  $$
  即用合成梯度$\hat\delta_{i+1}$计算$\delta_i$。由于合成梯度是可以立刻得出的，因此这次真的解除了更新锁定。要补充的细节是最后一层，即输出层由于没有下一层而能得到真实的梯度，所以$M_n$是收敛最快的。作者还提及这样的近似并不会导致更大的错误，效果仍然很好。

  **前馈、更新解耦**：

  上述模型只描述了合成梯度，实际上还可以预测输入解除正向锁定。下面的模型同时解除了三种锁定。

  ![image-20210209170659292.png](https://upload-images.jianshu.io/upload_images/16793245-c3f1824d24c8f7da.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

  **更新过程**：

  ![image-20210209172323461.png](https://upload-images.jianshu.io/upload_images/16793245-59a3c5bd1021bda0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

  首先，$M_{i+1}$预测$\hat\delta_i$，用于训练$f_i$；然后$M_{i+2}$预测$\hat\delta_{i+1}$，交给$f_{i+1}$计算梯度$\delta_i$，用于训练$M_{i+1}$；以此类推，训练所有的模型。

#### 1.5 Experiment

- 作者是在哪些数据集或者说场景下进行了测试？

  首先将DNIs应用于RNN，表明合成梯度扩展了RNN可以学习的时间相关性；其次，展示了如何使用合成梯度来联合训练一个分层的、双时间尺度的网络系统，以在网络之间传播错误信号；最后展示了DNIs允许异步更新前馈网络层的能力。

  **RNN**：

  作者使用的模型均为LSTM模型。

  在RNN中提到了三种任务：复制任务、重复复制任务和语言模型。复制任务就是给定长度N的字符序列+结束符，输出一样的序列。重复复制则是在末尾再添加一个输入R，表示重复次数。语言模型则是给出一个字符，预测下一个字符。作者对比了三个模型：BPTT（通过时间的反向传播）、DNI、DNI+Aux（合成梯度还将预测未来几步的梯度）。

  语言模型中使用了数据集Penn Treebank，是一个语料库。

  **FNN**：

  MNIST数据集、CIFAR-10数据集。使用的模型为FCN和CNN。

- 实验中的重要指标有哪些？

  **RNN**：

  实验结果如下：

  ![image-20210209163227809.png](https://upload-images.jianshu.io/upload_images/16793245-0407cb8c46c7c47b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

  ![image-20210209163853364.png](https://upload-images.jianshu.io/upload_images/16793245-9a39497578768b24.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

  表格中，T表示预测、使用的未来几步。Copy和Repeat Copy的任务指标是可成功建模的最大序列长度，越大越好；Penn Treebank表示语言模型的困惑度，越小越好。

  左图中，x轴表示模型使用的样本数，y轴表示模型解决的时间依赖性（不是很明白）（the time dependency level solved by the model – step changes in the time dependency indicate that a particular time dependency is deemed solved.）

  右图中，y轴表示BPC（bits per character，与困惑度相关。困惑度用于衡量一个语言模型在未见过的的字符串S上的表现）。还采用了Early stopping防止过拟合。

  BPC计算公式如下：
  $$
  \begin{aligned}
  b p c(s t r i n g)=\frac{1}{T} \sum_{t=1}^{T} H\left(P_{t}, \hat{P}_{t}\right) &=-\frac{1}{T} \sum_{t=1}^{T} \sum_{c=1}^{n} P_{t}(c) \log _{2} \hat{P}_{t}(c) \\
  &=-\frac{1}{T} \sum_{t=1}^{T} \log _{2} \hat{P}_{t}\left(x_{t}\right)
  \end{aligned}
  $$
  **FNN**：

  实验结果如下：

  ![image-20210209170908343.png](https://upload-images.jianshu.io/upload_images/16793245-1cae0ceb5a70000f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

  只使用了错误率一个指标。x轴为$p_\text{update}$，表示更新概率：每一次正向通过（forward pass）后，反向通过（backward pass）的概率；y轴为错误率；不同的颜色代表不同的迭代轮次。cDNI是指计算合成梯度时还加入了标签的DNI。

  左图为只使用了模型$M$合成梯度，右图还使用了模型$I$预测输入。

  可以看到cDNI性能要优秀很多，即使更新概率只有5%也能收敛到很好的效果（迭代次数500k）。

- 文章提出的方法在哪些指标上表现好？在哪些指标上表现不好？

  **RNN**：

  RNN的实验中，指标全面碾压。实验表明了DNI能够加深RNN的预测深度、甚至预测更好的结果。说明了DNI可以应用于RNN中，在性能不变/更好的前提下还能带来异步训练等好处。

  **FNN**：

  论文正文没有给出BP的数据，是在附录给出的。结果如下：

  <img src="https://upload-images.jianshu.io/upload_images/16793245-9acf32f4d7e7712a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" alt="image-20210209172944420.png" style="zoom:50%;" />

  

  上图为经过500k迭代后的最终测试结果对比；下图为训练过程中的测试误差对比。

  可以看到，无论是DNI还是cDNI的优势都不是特别大。而且DNI在网络变深时性能会明显下降；cDNI则表现好一点。值得注意的是还和不使用BP算法进行对比，说明DNI其实也没那么差。

#### 1.6 Conclusion

- 这篇论文最大的贡献是什么？

  DNI应用在RNN可以带来比较好的效果，加快收敛；允许前馈网络进行分布式、异步、非顺序、偶尔进行训练。

  这是第一次神经网络模块被解耦，更新锁定被打破。这个重要的结果打开了令人兴奋的探索之路——包括改进这里列出的基础，以及应用到模块化、解耦和异步模型架构。

- 论文中的方法还存在什么问题？

  论文中的方法如果应用到FNN，在最终结果上与BP算法还是有一些差距。

### 2 论文想要解决的问题？

#### 2.1 背景是什么？

BP算法里，网络的所有层、模块，都是锁定的，在它们可以被更新之前，它们必须等待网络的其余部分执行正向传播和反向传播误差。

#### 2.2 之前的方法存在哪些问题

之前的方法都无法打破更新锁定。

#### 2.3 输入和输出是什么？

提出的模型$M_{i+1}$输入是层$f_{i}$的输出，输出是层$f_i$的梯度；模型$I_i$的输入是网络输入$x$，输出是层$f_i$的输入。

### 3 论文研究的是否是一个新问题

不是一个新问题，研究解耦网络/打破三种锁定很早就开始了。这个问题如果成功解决可以让网络实现异步更新。

### 4 论文试图验证的科学假设

每一层的梯度可以通过另一个网络去预测；甚至包括输入也可以预测。

### 5 相关的关键人物与工作

#### 5.1 之前存在哪些相关的工作

消除反向传播但仍保留反向、更新锁定（target prop）；有的只能消除反向锁定而不消除更新锁定（REINFORCE、Kickback、Policy Gradient Coagent Networ）；有的是不可扩展的，要求优化器具有网络状态的全局知识（Real-Time Recurrent Learning、approximations）；Taylor、Carreira-Perpin an & Wang的工作允许在没有反向传播的情况下并行训练层，但在实践中不能扩展到更复杂和通用的网络架构。

#### 5.2 本文是对哪个工作进行的改进

相较之下，本篇论文通过将各层之间的相互作用定义为DNI的局部通信问题，以此消除了学习系统对全局知识的需要。最接近作者的工作的是“用于梯度上升的值函数”和“训练神经网络的critics”。

#### 5.3 这个领域的关键研究者

Baxter, J. and Bartlett, P. L.

Schmidhuber, J¨urgen.

Czarnecki

### 6 论文提出的解决方案的关键

提出DNI，用另一个神经网络拟合梯度、层输入。

### 7 论文的解决方案有完备的理论证明吗

关键公式：
$$
\begin{aligned}
\frac{\partial L}{\partial \theta_{i}} &=f_{\text {Bprop }}\left(\left(h_{i}, x_{i}, y_{i}, \theta_{i}\right), \ldots\right) \frac{\partial h_{i}}{\partial \theta_{i}} \\
& \simeq \hat{f}_{\text {Bprop }}\left(h_{i}\right) \frac{\partial h_{i}}{\partial \theta_{i}}
\end{aligned}
$$

### 8 实验设计

#### 8.1用到了哪些数据集

Penn Treebank语料库、MNIST手写数字数据集、CIFAR-10数据集。

#### 8.2与什么算法进行了比较

与传统BP算法、BPTT算法对比。

#### 8.3评价指标是什么

BPC、最大成功预测长度、错误率

#### 8.4有没有什么独特的实验实验设计？

在RNN的实验中，还测试了复制任务和重复复制任务。

以及通过更新概率来验证网络的随机、异步更新性。

### 9 实验支撑

#### 9.1 论文的数据集哪里获取

Penn Treebank语料库：在tf中可以通过`from tensorflow.models.rnn.ptb import reader`获取，或者[网址](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)下载。

#### 9.2 源代码哪里可以获取

GitHub上一个用pytorch实现的[代码](https://github.com/koz4k/dni-pytorch)。

#### 9.3 关键代码的讲解

```python
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init

from contextlib import contextmanager
from functools import partial


class UnidirectionalInterface(torch.nn.Module):
    """Basic `Interface` for unidirectional communication.
    Can be used to manually pass `messages` with methods `send` and `receive`.
    基接口，用于层间通信。用方法send和receive通信。
    Args:
        synthesizer: `Synthesizer` to use to generate `messages`.
    属性：
    	synthesizer：合成器，用于生成消息
    """

    def __init__(self, synthesizer):
        super().__init__()

        self.synthesizer = synthesizer

    def receive(self, trigger):
        """Synthesizes a `message` based on `trigger`.
        Detaches `message` so no gradient will go through it during the
        backward pass.
        基于触发器trigger生成消息。分离消息从而在反向通过时不会有梯度经过。
        Args:
            trigger: `trigger` to use to synthesize a `message`.
        Returns:
            The synthesized `message`.
        """
        return self.synthesizer(
            trigger, _Manager.get_current_context()
        ).detach()

    def send(self, message, trigger):
        """Updates the estimate of synthetic `message` based on `trigger`.
        Synthesizes a `message` based on `trigger`, computes the MSE between it
        and the input `message` and backpropagates it to compute its gradient
        w.r.t. `Synthesizer` parameters. Does not backpropagate through
        `trigger`.
        Args:
            message: Ground truth `message` that should be synthesized based on
                `trigger`.
            trigger: `trigger` that the `message` should be synthesized based
                on.
        """
        synthetic_message = self.synthesizer(
            trigger.detach(), _Manager.get_current_context()
        )
        loss = F.mse_loss(synthetic_message, message.detach())
        _Manager.backward(loss)


class ForwardInterface(UnidirectionalInterface):
    """`Interface` for synthesizing activations in the forward pass.
    Can be used to achieve a forward unlock. It does not make too much sense to
    use it on its own, as it breaks backpropagation (no gradients pass through
    `ForwardInterface`). To achieve both forward and update unlock, use
    `BidirectionalInterface`.
    Args:
        synthesizer: `Synthesizer` to use to generate `messages`.
    """

    def forward(self, message, trigger):
        """Synthetic forward pass, no backward pass.
        Convenience method combining `send` and `receive`. Updates the
        `message` estimate based on `trigger` and returns a synthetic
        `message`.
        Works only in `training` mode, otherwise just returns the input
        `message`.
        Args:
            message: Ground truth `message` that should be synthesized based on
                `trigger`.
            trigger: `trigger` that the `message` should be synthesized based
                on.
        Returns:
            The synthesized `message`.
        """
        if self.training:
            self.send(message, trigger)
            return self.receive(trigger)
        else:
            return message


class BackwardInterface(UnidirectionalInterface):
    """`Interface` for synthesizing gradients in the backward pass.
    Can be used to achieve an update unlock.
    Args:
        synthesizer: `Synthesizer` to use to generate gradients.
    """

    def forward(self, trigger):
        """Normal forward pass, synthetic backward pass.
        Convenience method combining `backward` and `make_trigger`. Can be
        used when we want to backpropagate synthetic gradients from and
        intercept real gradients at the same `Variable`, for example for
        update decoupling feed-forward networks.
        Backpropagates synthetic gradient from `trigger` and returns a copy of
        `trigger` with a synthetic gradient update operation attached.
        Works only in `training` mode, otherwise just returns the input
        `trigger`.
        Args:
            trigger: `trigger` to backpropagate synthetic gradient from and
                intercept real gradient at.
        Returns:
            A copy of `trigger` with a synthetic gradient update operation
            attached.
        """
        if self.training:
            self.backward(trigger)
            return self.make_trigger(trigger.detach())
        else:
            return trigger

    def backward(self, trigger, factor=1):
        """Backpropagates synthetic gradient from `trigger`.
        Computes synthetic gradient based on `trigger`, scales it by `factor`
        and backpropagates it from `trigger`.
        Works only in `training` mode, otherwise is a no-op.
        Args:
            trigger: `trigger` to compute synthetic gradient based on and to
                backpropagate it from.
            factor (optional): Factor by which to scale the synthetic gradient.
                Defaults to 1.
        """
        if self.training:
            synthetic_gradient = self.receive(trigger)
            _Manager.backward(trigger, synthetic_gradient.data * factor)

    def make_trigger(self, trigger):
        """Attaches a synthetic gradient update operation to `trigger`.
        Returns a `Variable` with the same `data` as `trigger`, that during
        the backward pass will intercept gradient passing through it and use
        this gradient to update the `Synthesizer`'s estimate.
        Works only in `training` mode, otherwise just returns the input
        `trigger`.
        Returns:
            A copy of `trigger` with a synthetic gradient update operation
            attached.
        """
        if self.training:
            return _SyntheticGradientUpdater.apply(
                trigger,
                self.synthesizer(trigger, _Manager.get_current_context())
            )
        else:
            return trigger


class _SyntheticGradientUpdater(torch.autograd.Function):

    @staticmethod
    def forward(ctx, trigger, synthetic_gradient):
        (_, needs_synthetic_gradient_grad) = ctx.needs_input_grad
        if not needs_synthetic_gradient_grad:
            raise ValueError(
                'synthetic_gradient should need gradient but it does not'
            )

        ctx.save_for_backward(synthetic_gradient)
        # clone trigger to force creating a new Variable with
        # requires_grad=True
        return trigger.clone()

    @staticmethod
    def backward(ctx, true_gradient):
        (synthetic_gradient,) = ctx.saved_variables
        # compute MSE gradient manually to avoid dependency on PyTorch
        # internals
        (batch_size, *_) = synthetic_gradient.size()
        grad_synthetic_gradient = (
            2 / batch_size * (synthetic_gradient - true_gradient)
        )
        return (true_gradient, grad_synthetic_gradient)


class BidirectionalInterface(torch.nn.Module):
    """`Interface` for synthesizing both activations and gradients w.r.t. them.
    Can be used to achieve a full unlock.
    Args:
        forward_synthesizer: `Synthesizer` to use to generate `messages`.
        backward_synthesizer: `Synthesizer` to use to generate gradients w.r.t.
            `messages`.
    """

    def __init__(self, forward_synthesizer, backward_synthesizer):
        super().__init__()

        self.forward_interface = ForwardInterface(forward_synthesizer)
        self.backward_interface = BackwardInterface(backward_synthesizer)

    def forward(self, message, trigger):
        """Synthetic forward pass, synthetic backward pass.
        Convenience method combining `send` and `receive`. Can be used when we
        want to `send` and immediately `receive` using the same `trigger`. For
        more complex scenarios, `send` and `receive` need to be used
        separately.
        Updates the `message` estimate based on `trigger`, backpropagates
        synthetic gradient from `message` and returns a synthetic `message`
        with a synthetic gradient update operation attached.
        Works only in `training` mode, otherwise just returns the input
        `message`.
        """
        if self.training:
            self.send(message, trigger)
            return self.receive(trigger)
        else:
            return message

    def receive(self, trigger):
        """Combination of `ForwardInterface.receive` and
        `BackwardInterface.make_trigger`.
        Generates a synthetic `message` based on `trigger` and attaches to it
        a synthetic gradient update operation.
        Args:
            trigger: `trigger` to use to synthesize a `message`.
        Returns:
            The synthesized `message` with a synthetic gradient update
            operation attached.
        """
        message = self.forward_interface.receive(trigger)
        return self.backward_interface.make_trigger(message)

    def send(self, message, trigger):
        """Combination of `ForwardInterface.send` and
        `BackwardInterface.backward`.
        Updates the estimate of synthetic `message` based on `trigger` and
        backpropagates synthetic gradient from `message`.
        Args:
            message: Ground truth `message` that should be synthesized based on
                `trigger` and that synthetic gradient should be backpropagated
                from.
            trigger: `trigger` that the `message` should be synthesized based
                on.
        """
        self.forward_interface.send(message, trigger)
        self.backward_interface.backward(message)


class BasicSynthesizer(torch.nn.Module):
    """Basic `Synthesizer` based on an MLP with ReLU activation.
    Args:
        output_dim: Dimensionality of the synthesized `messages`.
        n_hidden (optional): Number of hidden layers. Defaults to 0.
        hidden_dim (optional): Dimensionality of the hidden layers. Defaults to
            `output_dim`.
        trigger_dim (optional): Dimensionality of the trigger. Defaults to
            `output_dim`.
        context_dim (optional): Dimensionality of the context. If `None`, do
            not use context. Defaults to `None`.
    """

    def __init__(self, output_dim, n_hidden=0, hidden_dim=None,
                 trigger_dim=None, context_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = output_dim
        if trigger_dim is None:
            trigger_dim = output_dim

        top_layer_dim = output_dim if n_hidden == 0 else hidden_dim

        self.input_trigger = torch.nn.Linear(
            in_features=trigger_dim, out_features=top_layer_dim
        )

        if context_dim is not None:
            self.input_context = torch.nn.Linear(
                in_features=context_dim, out_features=top_layer_dim
            )
        else:
            self.input_context = None

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=(
                    hidden_dim if layer_index < n_hidden - 1 else output_dim
                )
            )
            for layer_index in range(n_hidden)
        ])

        # zero-initialize the last layer, as in the paper
        if n_hidden > 0:
            init.constant(self.layers[-1].weight, 0)
        else:
            init.constant(self.input_trigger.weight, 0)
            if context_dim is not None:
                init.constant(self.input_context.weight, 0)

    def forward(self, trigger, context):
        """Synthesizes a `message` based on `trigger` and `context`.
        Args:
            trigger: `trigger` to synthesize the `message` based on. Size:
                (`batch_size`, `trigger_dim`).
            context: `context` to condition the synthesizer. Ignored if
                `context_dim` has not been specified in the constructor. Size:
                (`batch_size`, `context_dim`).
        Returns:
            The synthesized `message`.
        """
        last = self.input_trigger(trigger)

        if self.input_context is not None:
            last += self.input_context(context)

        for layer in self.layers:
            last = layer(F.relu(last))

        return last


@contextmanager
def defer_backward():
    """Defers backpropagation until the end of scope.
    Accumulates all gradients passed to `dni.backward` inside the scope and
    backpropagates them all in a single `torch.autograd.backward` call.
    Use it and `dni.backward` whenever you want to backpropagate multiple times
    through the same nodes in the computation graph, for example when mixing
    real and synthetic gradients. Otherwise, PyTorch will complain about
    backpropagating more than once through the same graph.
    Scopes of this context manager cannot be nested.
    """
    if _Manager.defer_backward:
        raise RuntimeError('cannot nest defer_backward')
    _Manager.defer_backward = True

    try:
        yield

        if _Manager.deferred_gradients:
            (variables, gradients) = zip(*_Manager.deferred_gradients)
            torch.autograd.backward(variables, gradients)
    finally:
        _Manager.reset_defer_backward()


@contextmanager
def synthesizer_context(context):
    """Conditions `Synthesizer` calls within the scope on the given `context`.
    All `Synthesizer.forward` calls within the scope will receive `context`
    as an argument.
    Scopes of this context manager can be nested.
    """
    _Manager.context_stack.append(context)
    yield
    _Manager.context_stack.pop()


class _Manager:

    defer_backward = False
    deferred_gradients = []
    context_stack = []

    @classmethod
    def reset_defer_backward(cls):
        cls.defer_backward = False
        cls.deferred_gradients = []

    @classmethod
    def backward(cls, variable, gradient=None):
        if gradient is None:
            gradient = _ones_like(variable.data)

        if cls.defer_backward:
            cls.deferred_gradients.append((variable, gradient))
        else:
            variable.backward(gradient)

    @classmethod
    def get_current_context(cls):
        if cls.context_stack:
            return cls.context_stack[-1]
        else:
            return None


"""A simplified variant of `torch.autograd.backward` influenced by
`defer_backward`.
Inside of `defer_backward` scope, accumulates passed gradient to backpropagate
it at the end of scope. Outside of `defer_backward`, backpropagates the
gradient immediately.
Use it and `defer_backward` whenever you want to backpropagate multiple times
through the same nodes in the computation graph.
Args:
    variable: `Variable` to backpropagate the gradient from.
    gradient (optional): Gradient to backpropagate from `variable`. Defaults
        to a `Tensor` of the same size as `variable`, filled with 1.
"""
backward = _Manager.backward


def _ones_like(tensor):
    return tensor.new().resize_(tensor.size()).fill_(1)
```

#### 9.4 补充

这篇论文的附录很长，占了全文的一半。附录中包含的内容有：

* 详细的理论推导过程
* FNN实验中，no BP、BP、DNI、cDNI的数据对比和分析
* 对比了随着层数增加，DNI和cDNI的效果。作者提到在21层的网络中，CIFAR-10数据集cDNI和BP到达了相同的效果2%。
* 还记录了每一层DNI的指标变化（信号错误率，余弦相似度和梯度误差）
* 给出了cDNI、DNI的实现细节
* 给出了实验使用的所有模型的细节

### 10 实验结果是否验证了科学假设？

已验证。通过另一个网络可以模拟出需要的梯度。

### 11 论文最大的贡献

DNI应用在RNN可以带来比较好的效果，加快收敛；允许前馈网络进行分布式、异步、非顺序、偶尔进行训练。

这是第一次神经网络模块被解耦，更新锁定被打破。这个重要的结果打开了令人兴奋的探索之路——包括改进这里列出的基础，以及应用到模块化、解耦和异步模型架构。

### 12 论文的不足之处

论文中的方法如果应用到FNN，在最终结果上与BP算法还是有一些差距。

#### 12.1 这篇论文之后的工作有哪些其他的改进

作者提到混合真实梯度和合成梯度这种用法。虽然这样会重新回到那三种锁定，但是可能会带来其他意想不到的好处。这个已经在强化学习领域进行了研究（1998年）。

#### 12.2 你觉得可以对这篇论文有什么改进

感觉写的挺详细了。

### 13 重要的相关论文

Czarnecki, W M, Swirszcz, G, Jaderberg, M, Osindero, S, Vinyals, O, and Kavukcuoglu, K. Understanding synthetic gradients and decoupled neural interfaces. arXiv preprint, 2017. [连接](http://proceedings.mlr.press/v70/czarnecki17a/czarnecki17a.pdf)

这篇论文更详细、深度地分析DNI和合成梯度.

### 14 不懂之处

不懂之处是RNN那块的推导。

### 15 参考博客

https://www.codercto.com/a/37654.html
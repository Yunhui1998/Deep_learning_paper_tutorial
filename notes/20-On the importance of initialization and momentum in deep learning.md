## 20 Momentum: On the importance of initialization and momentum in deep learning

关于初始化和动量在深度学习中的重要性

### 1  论文的结构(简要概括)

#### 1.1  Abstract

- 作者想解决什么问题？

  利用一种较为简单的一阶优化方法，来训练DNNs和RNNs（具有长期依赖问题），提升网络性能。

- 作者通过什么理论/模型来解决这个问题？

  利用设计良好的初始化（本文主要是调参）和一种特殊的缓慢增加的动量参数。

- 作者给出的答案是什么？

  简单的一阶优化方法是可行的。Momentum和初始化都至关重要，以前尝试从随机初始化训练深度和循环神经网络的尝试可能由于初始化方案不佳而失败。此外，仔细调整动量方法，可以和用于处理深度和循环网络训练目标中的曲率问题相结合，也不需要复杂的二阶方法。

#### 1.2 Introduction

- 作者为什么研究这个课题？

  DNNs和RNNs难以训练，具有长期依赖性的问题（即经过许多阶段梯度传播后，梯度倾向于消失或者爆炸。循环神经网络涉及到多次相同函数的组合，这些组合导致极短的非线性）。业界提出了几种优化方法，但简单的一阶优化方法仍被认为是不可行的。

- 目前这个课题的研究进行到了哪一阶段？存在哪些缺陷？作者是想通过本文解决哪个问题？

  使用一种类似动量的一阶优化方法，并（必须）结合精心设计的初始化动量常数，来发现一阶动量方法真正的有效性。之后开发了一种类似动量的HF法改进实验结果。


#### 1.3 Related work

- 和作者这篇论文相关的工作有哪些？

  对比算法：当时较突出的***Hessian-free Optimization***(HF)的集群牛顿方法对DNNs,RNNs的训练效果较好，但比较复杂，本文最后开发了一种结合的动量的HF算法。

- 之前工作的优缺点是什么？

  之前人们认为简单的一阶动量优化是不切实际的，因为在渐近局部收敛时必须精细的搜索，动量在这方面将丧失速度的优化性，最后的准确度也较低。

- 作者主要是对之前的哪个工作进行改进？

  但在实践中，发生在精细的局部收敛开始之前，有一个收敛的暂态阶段(Darken&Moody, 1993)，似乎对优化深层神经网络更重要。作者希望在这个粗糙部分可以用一阶方法加快训练。

#### 1.4  Theoretical Analysis

- 作者是用什么理论证明了自己的方法在理论上也是有保障的？

  用正定二次型的方法，理论推导一阶动量的抗震荡性，这样加大动量参数μ可以加速梯度下降，并且性能不至于变差。

#### 1.5 Experiment

- 作者是在哪些数据集或者说场景下进行了测试？文章提出的方法在哪些指标上表现好？在哪些指标上表现不好？

  利用Hinton& Salakhutdinov（2006）中描述的三个深度自动编码器测试框架，指标为错误率。发现运用简单的一阶动量测试，性能可逼近当时最好的优化算法HF。但对稀疏特征应用性较差，需要手动设置初始化。

- 在实验的设置过程中作者有没有提到自己用到了什么trick？

  根据经验，学习率 $\varepsilon$ 的选择：范围为{0.05,0.01,0.005,0.001,0.0005,0.0001}，每次实验使用结果最好的学习率。
  
  动量系数$μ_{max}$的取值为{0.999, 0.995, 0.99, 0.9, 0} 。

#### 1.6 Conclusion

- 这篇论文最大的贡献是什么？

  当结合精心选择的初始化方案和各种形式的动量为基础，虽然优化后的算法和当时最先进的HF仍存在一定差距，但具有更加简单的一阶形式。

- 为什么有效？

  在深度学习问题中，学习的最后需要精细收敛的阶段并不像最初的过渡阶段那么长，也不那么重要。(Darken & Moody, 1993)。即过渡段可以粗糙搜索，利用一阶动量加快速度。

- 论文中的方法还存在什么问题？

  论文提出的动量法必须辅佐合适的初始化，否则动量法带来的性能优化效果并不明显。这也是论文标题为什么要强调初始化的原因，两者缺一不可。

### 2 论文想要解决的问题？

#### 2.1 背景是什么？

DNNs和RNNs难以训练，存在在数据集上的长期依赖的问题。有希望解决这些问题的算法过于复杂，而由于梯度消失的问题，被认为不可能找到简单有效的一阶优化方法。

### 3 论文研究的是否是一个新问题

不是，之前已有类似的优化方法，但对一阶简单优化方法仍认为不可行的（性能不够好）。

### 4 论文试图验证的科学假设

能够找到一种简单的一阶优化方法，并具有良好的性能。

### 5 相关的关键人物与工作

#### 5.1 之前存在哪些相关的工作

- 贪婪的分层预训练***greedy layerwise pre-training***（Hinton.2006)，先使用辅助目标按顺序训练DNN层，然后使用随机梯度下降(SGD)等标准优化方法对整个网络进行微调。
- ***Hessian-free Optimization***(HF)的集群牛顿方法能够在不使用预训练的情况下从某些随机初始化训练DNNs。以上两种方法均存在过于复杂的问题。
- HF方法也可以有效地训练RNNs处理具有长期依赖性的人工问题，并获得了较低的误差。比起动量法更灵活的应对系数问题（因为HF类似二阶加速度），但除了较为复杂，缺少指数加权平均数也使算法占内存，不可中途停止。

#### 5.3 这个领域的关键研究者

Ilya Sutskever，三巨头之一Hinton的学生。

![image-20210202170723349](D:\深度学习\论文\20-On the importance of initialization and momentum in deep learning.assets\image-20210202170723349.png)

参与论文：

AlexNet：[Imagenet classification with deep convolutional neural networks](javascript:void(0))

利用端到端的(End *To* End)模型处理序列到序列的任务：[Sequence to sequence learning with neural networks](javascript:void(0))



### 6 论文提出的解决方案的关键

#### 6.1 经典动量法classical momentum(CM)

经典动量classical momentum(CM)，是一种加速梯度下降的技术，它在迭代过程中不断减少目标的方向上积累速度矢量。给定要最小化的目标函数![img](https://img-blog.csdnimg.cn/20190313124541642.png)，经典动量为：

![img](https://img-blog.csdnimg.cn/20190313124528545.png)

![img](https://img-blog.csdnimg.cn/20190313124600285.png)是学习速率，![img](https://img-blog.csdnimg.cn/2019031312461769.png)是动量系数，而![img](https://img-blog.csdnimg.cn/2019031312470074.png)是![img](https://img-blog.csdnimg.cn/20190313124650987.png)的梯度。

#### 6.2 NAG一阶优化

Nesterov’s Accelerated Gradient(abbrv: NAG, Nesterov, 1983)一直是凸优化界关注的热点问题(Cotter et al., 2011; Lan, 2010)。类似动量法，NAG，比梯度下降法具有更好的收敛速度保证，特别是对于一般光滑(非强)凸函数和确定性梯度。

NAG更新可以写为：

![img](https://img-blog.csdnimg.cn/20190313125241715.png)

CM和NAG的计算利用指数平均数$v_t$，基于新的速度来应用梯度，修正了先前的速度矢量衰退。

对比CM计算梯度更新利用当前的速度$\theta_t$，NAG更新利用的是$\theta_t+μv_t$。这种看似友善的差异似乎能让NAG以更快、更灵敏的方式改变v，让它在许多情况下比CM表现得更稳定，尤其是在值更高的动量系数$μ$下。

由于$μv_t$的结果立即改变目标函数f。如果$μv_t$是一个很差的更新，则NAG的梯度$\nabla f\left(\theta_{t}+\mu v_{t}\right)$将比$\nabla f\left(\theta_{t}\right)$更快的校正，从而提供一个比CM更大、更及时的修正。图1从几何上说明了这一现象。

![img](https://img-blog.csdnimg.cn/20190313130222702.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNzk5MzI1MQ==,size_16,color_FFFFFF,t_70)

之后将CM和NAG应用于正定二次目标$q(x)=x^{\top} A x / 2+b^{\top}$，从理论上证明了NAG对高曲率本征方向使用较小的动量，这可以防止振荡(或差异)，从而允许使用更大的µ，而为什么CM使用更大的µ效果会下降。这一点可从本文实验结果中看出。

### 7 论文的解决方案有完备的理论证明吗

动量法的公式可以看出加速了梯度下降。接着运用数学推导，证明动量法可以使用更大的 μ 加快训练速度。

### 8 实验设计

#### 8.1用到了哪些数据集

利用Hinton& Salakhutdinov（2006）中描述的三个深度自动编码器问题。神经网络自编码器的任务是在其隐含层之一是低维的约束下重构自己的输入。这个“瓶颈”层充当原始输入的低维代码，类似于其他降维技术，如主成分分析（PCA）。这些自动编码器是一些已公布结果的深度神经网络，范围在7到11层之间，并已成为一个标准的算法基准测试问题。

#### 8.2与什么算法进行了比较

学习率 $\varepsilon$ 的取值为{0.05,0.01,0.005,0.001,0.0005,0.0001}，每次实验使用结果最好的学习率。

$μ_t$的更新公式：

![image.png](https://upload-images.jianshu.io/upload_images/24435792-8bc4cd248e826587.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

动量系数$μ_{max}$的取值为{0.999, 0.995, 0.99, 0.9, 0} 。对于在非强凸函数上使用$μ_t = 1−3/(t+5)$更新。

本文专注于优化和调参，以寻找合适的动量系数$μ_{max}$，并对比CM和NAG的性能。在优化算法方面则与当时流行的HF法对比。

***Hessian-free Optimization***(HF)的集群牛顿方法能够在不使用预训练的情况下从某些随机初始化训练DNNs，并且能够在考虑的各种自动编码任务中获得较低的误差。Hinton&Salakhutdinov (2006)。

#### 8.3 结果展示

![image-20210202090424135](D:\深度学习\论文\20-On the importance of initialization and momentum in deep learning.assets\image-20210202090424135.png)

该表报告了$μ_{max}$和动量类型(NAG, CM)的每一种组合的问题的平方误差。当μ为0时相当于SGD，可以发现NAG的性能高于CM，且对增大μ的性能提升较为明显。

然而作者提出最后的1000次迭代中，在不降低学习速率的情况下，将μ减小到0.9是有益的。如下表所示，表中显示了动量系数降低前后的训练平方误差。在初级(暂态)学习阶段使用最佳动量和学习速率，而在接近收敛的后期选择降低动量参数。

![image-20210202090756096](D:\深度学习\论文\20-On the importance of initialization and momentum in deep learning.assets\image-20210202090756096.png)

解释：一个较大的μ允许朝有用的进展以及不常更改方向加快更新。然而，在低曲率（稀疏特征区 ）方向上的这一进展是一个新的参数区域，更接近最优（在凸目标的情况下），或更高质量的局部极小（在非凸优化的情况下），需要更精细小心的收敛。减少μ的值可以达到这种效果。作者主要关注更简单的一阶方法，因此没有讨论二阶加速度的问题。

二阶方法放大了在低曲率方向上的步骤，但是当时已有的方法并没有积累变化（即指数加权平均数的递归方法），而是通过相关曲率的倒数对每个曲率矩阵特征方向上的更新重新加权。

#### 8.4 Recurrent Neural Networks

- 随机初始化设置

使用Martens(2010)中描述的稀疏初始化技术(SI)进行初始化。每个随机单元与上一层随机选取的15个单元相连，这些单元的权值由一个单位高斯抽取，并将偏差设为零，让它们对输入的响应在本质上更加“多样化”。

- RNNs训练

  RNNs表现出相当长的时间依赖性，但有动量加速的SGD可以成功地训练（但需要较小的学习率）。之前人们认为，由于各种困难，比如梯度消失/爆炸，几乎不可能用一阶方法在这些数据集上成功训练rnn (Bengio et al.， 1994)。

  ![img](https://img-blog.csdnimg.cn/20190313132109816.png)

  对4种不同的随机种子，探究$μ_0$和动量类型(NAG, CM)的每一种组合的不同问题。可看出，在NAG中，$μ_0$值越大，效果越好，而在CM中则没有，这可能是由于NAG对较大$μ_0$的耐受能力(见6.2节讨论)。

#### 8.5 Momentum and HF 

截尾牛顿法，以Martens(2010)的HF法为例，是一种通过线性共轭梯度算法(CG)对目标的局部二次模型进行优化，与梯度相比，HF可以在曲率较低的某些方向上自然加速。它在迭代过程中积累信息，但不能中途停止。相反，动量方法保存的信息可以在任意数量的迭代中通知新的更新，也就是指数加权平均数带来的好处。

文章提议可以将Momentum 和 HF 结合使用，并在附录描述了一些实验的细节。 

### 9 实验支撑

#### 9.1 论文的数据集哪里获取

minist数据集：http://yann.lecun.com/exdb/mnist/，包括在三个深度自动编码器问题内，但没有找到开源的数据集。

#### 9.2 相关代码

**代码导入数据集**

- **导入库函数**

```python
import numpy as np
import matplotlib.pyplot as plt
import math

import opt_utils #需要用到的数据包
import testCase  #数据包

plt.rcParams['figure.figsize'] = (7.0, 4.0) #设置图像大小等信息
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

- **加载数据集**

```python
train_X, train_Y = opt_utils.load_dataset(is_plot=True)
```

训练集分布情况如图所示：

![1.png](https://upload-images.jianshu.io/upload_images/24435917-97178d598bd091ad.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- **构造神经网络模型，实现momentum算法**

```python
def model(X,Y,layers_dims,learning_rate=0.0007,
          mini_batch_size=64,beta=0.9,beta1=0.9,
          num_epochs=60000,print_cost=True,is_plot=True):
    
    """
    可以运行在不同优化器模式下的3层神经网络模型。
    
    参数：
        X - 输入数据，维度为（点阵坐标，输入的数据集里面样本数量）2*n
        Y - 与X对应的标签
        layers_dims - 包含层数和节点数量的列表
        learning_rate - 学习率
        mini_batch_size - 每个小批量数据集的大小
        beta - 用于动量优化的一个超参数
        beta1 - 用于计算梯度后的指数衰减的估计的超参数
        num_epochs - 整个训练集的迭代次数
        print_cost - 是否打印误差值，每遍历1000次数据集打印一次，但是每100次记录一个误差值，又称每1000代打印一次
        is_plot - 是否绘制出曲线图
        
    返回：
        parameters - 包含了学习后的参数
        
    """
    L = len(layers_dims)
    costs = []
    t = 0 #每学习完一个minibatch就增加1
    seed = 10 #随机种子
    
    #初始化参数
    parameters = opt_utils.initialize_parameters(layers_dims)
    #使用动量
    v = initialize_velocity(parameters) 

    
    #开始学习
    for i in range(num_epochs):
        #定义随机 minibatches,我们在每次遍历数据集之后增加种子以重新排列数据集，使每次数据的顺序都不同
        seed = seed + 1
        minibatches = random_mini_batches(X,Y,mini_batch_size,seed)
        
        for minibatch in minibatches:
            #选择一个minibatch
            (minibatch_X,minibatch_Y) = minibatch
            
            #前向传播
            A3 , cache = opt_utils.forward_propagation(minibatch_X,parameters)
            
            #计算误差
            cost = opt_utils.compute_cost(A3 , minibatch_Y)
            
            #反向传播
            grads = opt_utils.backward_propagation(minibatch_X,minibatch_Y,cache)
            
            #更新参数
            parameters, v = update_parameters_with_momentun(parameters,grads,v,beta,learning_rate)

        #记录误差值
        if i % 100 == 0:
            costs.append(cost)
            #是否打印误差值
            if print_cost and i % 1000 == 0:
                print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))
    #是否绘制曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()
    
    return parameters
```

#### 9.2.1 mini-batch的实现

```python
def random_mini_batches(X,Y,mini_batch_size,seed):
    """
    从（X，Y）中创建一个随机的mini-batch列表
    
    参数：
        X - 输入数据，维度为(输入节点数量，样本的数量)
        Y - 对应的是X的标签，如【1 | 0】（蓝|红），维度为(1,样本的数量)
        mini_batch_size - 每个mini-batch的样本数量
        
    返回：
        mini-bacthes - 一个同步列表，维度为([mini_batch_X],[mini_batch_Y])，索引为对应的一个batch
    """
    
    np.random.seed(seed) #指定随机种子，不同种子数下生成的随机数组不同
    #此处目的为每次遍历数据集之后增加种子以重新排列数据集，使每次数据的顺序都不同
    m = X.shape[1] #行数，即样本数
    mini_batches = []
    
    #第一步：打乱顺序
    permutation = list(np.random.permutation(m)) #它会返回一个长度为m的随机数组，且里面的数是0到m-1
    shuffled_X = X[:,permutation]   #将每一列的数据按permutation的顺序来重新排列。
    shuffled_Y = Y[:,permutation].reshape((1,m))
        
    #第二步，分割
    num_complete_minibatches = math.floor(m / mini_batch_size) #把你的训练集分割成多少份。floor求整：请注意，如果值是99.99，那么返回值是99，剩下的0.99会被舍弃
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k+1)*mini_batch_size]

        mini_batch = (mini_batch_X,mini_batch_Y) #一个单独的mini-batch
        mini_batches.append(mini_batch) #在列表末尾添加新的对象。
    
    #如果训练集的大小刚好是mini_batch_size的整数倍，那么这里已经处理完了
    #如果训练集的大小不是mini_batch_size的整数倍，那么最后肯定会剩下一些，我们要把它处理了
    if m % mini_batch_size != 0:
        #获取最后剩余的部分
        mini_batch_X = shuffled_X[:,mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:,mini_batch_size * num_complete_minibatches:]
        
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches
```

#### 9.2.2 Momentum优化算法的实现

初始化速度

```python
def initialize_velocity(parameters):
    """
    初始化速度，velocity是一个字典：
        - keys: "dW1", "db1", ..., "dWL", "dbL" 
        - values:与相应的梯度/参数维度相同的值为零的矩阵。
    参数：
        parameters - 一个字典，包含了以下参数：
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
    返回:
        v - 一个字典变量，包含了以下参数：
            v["dW" + str(l)] = dWl的速度
            v["db" + str(l)] = dbl的速度
    
    """
    L = len(parameters) // 2 #神经网络的层数
    v = {}
    
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    
    return v

```

利用动量更新，影响梯度的方向：

```python
def update_parameters_with_momentun(parameters,grads,v,beta,learning_rate):
    """
    使用动量更新参数
    参数：
        parameters - 一个字典类型的变量，包含了以下字段：
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
        grads - 一个包含梯度值的字典变量，具有以下字段：
            grads["dW" + str(l)] = dWl
            grads["db" + str(l)] = dbl
        v - 包含当前速度的字典变量，具有以下字段：
            v["dW" + str(l)] = ...
            v["db" + str(l)] = ...
        beta - 超参数，动量，实数
        learning_rate - 学习率，实数
    返回：
        parameters - 更新后的参数字典
        v - 包含了更新后的速度变量
    """
    L = len(parameters) // 2 
    for l in range(L):
        #计算速度
        v["dW" + str(l + 1)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l + 1)]
        
        #更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
    
    return parameters,v

```

- **使用动量的梯度下降**

```python
layers_dims = [train_X.shape[0],5,2,1]
parameters = model(train_X, train_Y, layers_dims, beta=0.9,is_plot=True)
```

算法收敛情况和分类情况如图所示：

![31.png](https://upload-images.jianshu.io/upload_images/24435917-1d1c6caae738ec7c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![32.png](https://upload-images.jianshu.io/upload_images/24435917-37be9c1ec7f17cad.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 10 实验结果是否验证了科学假设？

是，动量法只要有合适的初始化，测试性能可以接近复杂的优化算法，证明简单的一阶优化方法是可行的。

### 11 论文最大的贡献

证明利用一阶动量法对优化网络训练的可行性，并且利用指数加权平均数的思想，比起原来的算法能够积累信息，递归的t+1步只和第t步有关。

### 12 论文的不足之处

#### 12.1 这篇论文之后的工作有哪些其他的改进

在此之后动量被广泛运用到算法中，比如带动量的适应性梯度算法（AdaGrad），以及文中的指数加权平均数的方法运用到二阶领域，两者结合下再创造了Adam算法。

#### 12.2你觉得可以对这篇论文有什么改进

即在最后迭代至靠近最优点时，采用人为地把动量参数μ降低来达到精细收敛的目的，以提高性能，因此本文中强调的初始化实际上是一种限制条件。

而同期可以应用在稀疏特征上的二阶加速度方法，缺少指数平均数的运用，在每一步中进行更新重新加权。

### 13 重要的相关论文

- 测试数据集，用自编码器对数据进行降维，揭开深度学习序幕的文章：[Hinton, G and Salakhutdinov, R. Reducing the dimensionality of data with neural networks. Science, 313:504-507, 2006.](https://www.cs.toronto.edu/~hinton/science.pdf)

- 本文的对比算法，HF在训练DNNs和RNNs上都获得了较小的误差[Martens, J. Deep learning via Hessian-free optimization. In Proceedings of the 27th International Conference on Machine Learning (ICML), 2010.](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)

- 本文理论依据，使用动量的有效性：[Darken, C. and Moody, J. Towards faster stochastic gradient search. Advances in neural information processing ystems, pp. 1009{1009, 1993.](https://proceedings.neurips.cc/paper/1991/file/e2230b853516e7b05d79744fbd4c9c13-Paper.pdf)

### 14 不懂之处

理论证明为什么NAG的抗震荡性比CM好，用到了一些高数知识。
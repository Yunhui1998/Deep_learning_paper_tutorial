## 10 ImageNet A Large-Scale Hierarchical Image Database

论文建立了ImageNet数据集，旨在用平均500-1000张清晰和全分辨率的图像来填充WordNet数据集同义词集的图片数据，来构建一个开源的大规模数据库。

### 1  论文的结构(简要概括)

#### 1.1  Abstract

- 作者想解决什么问题？

  随着互联网上图像数据的爆炸式增长，有可能建立更复杂、更强大的模型和算法。但是，**如何准确利用和组织此类数据仍然是一个关键问题**。

- 作者通过什么理论/模型来解决这个问题？

  ImageNet数据集**基于WordNet的结构**建立的大规模图像本体，旨在用平均500-1000张清晰和全分辨率的图像来填充WordNet的80,000个同义词集的大部分。

- 作者给出的答案是什么？

  ImageNet数据集具有大容量，高精度，开源的优点，具有优越的性能，可为计算机研究提供帮助。

#### 1.2 Introduction

- 作者为什么研究这个课题？

  数字时代带来了巨大的数据爆炸，通过利用这些图像，可以提出更复杂、更健壮的模型和算法，从而为用户提供更好的应用来索引、检索、组织这些数据并与之交互。

  但是究竟如何利用和组织这些数据是一个尚待解决的问题。

- 目前这个课题的研究进行到了哪一阶段？存在哪些缺陷？作者是想通过本文解决哪个问题？

  作者希望构建一个高质量数据库，来利用组织互联网上的图像数据，为计算机算法和模型的发展提供帮助。

  论文发表时还仅仅涵盖WordNet同义词集的〜10％，目标是在大约5万个同义词集中散布着大约5000万张清晰、多样化和全分辨率的图像。

  截至 2016 年，ImageNet 中含有超过 1500 万由人手工注释的图片网址，标签说明了图片中的内容，超过 2.2 万个类别。其中，至少有 100 万张里面提供了边框信息（bounding box）。

- 作者使用的理论是基于哪些假设？

  未使用理论。

#### 1.3 Related work

- 和作者这篇论文相关的工作有哪些？

  相关的数据集：WordNet语义数据集、标记良好的小型数据集(Caltech101/256 [8, 12], MSRC [22], PASCAL [7] etc.)、TinyImage、ESP数据集、LabelMe 、Lotus Hill数据集。

- 之前工作的优缺点是什么？

  以往的数据集有容量较小，精度不足，无层次结构，不开源等缺点。

  LabelMe 、Lotus Hill数据集提供详细的对象轮廓，这是ImageNet所缺少的，但有容量较小的缺点。

- 作者主要是对之前的哪个工作进行改进？

  对数据集容量，精度的改进，并基于WordNet的语义层次结构。

#### 1.4  Theoretical Analysis

- 作者是用什么理论证明了自己的方法在理论上也是有保障的？

  未使用理论，容量更大，分辨率更高，有层次结构的数据集性能会更好。

#### 1.5 Experiment

- 作者是在哪些数据集或者说场景下进行了测试？

  对比Caltech256探究提升数据集规模带来的改进。

  对比有人工清洗和没有，探究提升数据集分辨率带来的改进。

  对比层次结构和独立结构，探究层次结构的优点。

- 实验中的重要指标有哪些？

  验证ImageNet数据集的大规模，层次结构，高分辨率。

- 文章提出的方法在哪些指标上表现好？在哪些指标上表现不好？

  ImageNet数据集的大规模，层次结构，高分辨率的优点带来了算法性能的提升。

- 在实验的设置过程中作者有没有提到自己用到了什么trick？

  无

#### 1.6 Conclusion

- 这篇论文最大的贡献是什么？

  提供了大规模、高精度、有多样性和层次结构的数据集，可以为计算机视觉社区以及其他领域的研究人员提供机会。

- 论文中的方法还存在什么问题？

  论文发表时还未完成ImageNet数据集，并且作者仍为数据集中缺少边框、图像分割等信息。

- 作者觉得还可以怎么改进？

  完成ImageNet数据集，加入图像的新信息。

### 2 论文想要解决的问题？

#### 2.1 背景是什么？

互联网上图像数据的爆炸式增长有可能建立更复杂、更强大的模型和算法，以对图像和多媒体数据进行索引、检索、组织和交互。但是，**如何准确利用和组织此类数据仍然是一个关键问题**。

#### 2.2 之前的方法存在哪些问题

之前已有的数据集中：

- 小型数据集(Caltech101/256 [8, 12], MSRC [22], PASCAL [7] etc.)的数据量不足。
- TinyImage数据集在某些应用中已经取得成功，但是高水平的噪声和低分辨率图像使其不太适合通用算法的开发、训练和评估。
- ESP数据集拥有语义层次结构，但整个语义层次结构的分布不够均衡，语义有歧义，导致ESP数据的准确性和实用性可能会受到影响。并且数据集不开源。
- LabelMe 和Lotus Hill数据集为视觉界提供了补充资源，提供了对象的轮廓和位置。ImageNet缺少此类信息，但类别的数量和每个类别的图像数量已经远远超过了这两个数据集。

#### 2.3 输入和输出是什么？

ImageNet旨在用平均500-1000张清晰和全分辨率的图像来填充WordNet数据集同义词集的图片数据，来构建一个开源的大规模数据库。

### 3 论文研究的是否是一个新问题

不是，之前已有类似的图片分类数据库。ImageNet的标签类别是基于WordNet的，研究者希望ImageNet能够为WordNet层次结构中的大多数同义词提供数千万个干净整理的图像(cleanly sorted images)，相比于当前的其他图像数据集的规模要大得多，并且更多样、更准确。

### 4 论文试图验证的科学假设

构建的高精度，大容量ImageNet数据集是合理的，并且具有强大的性能。

### 5 相关的关键人物与工作

#### 5.1 之前存在哪些相关的工作

之前已有的数据集中：

- 标记良好的小型数据集(Caltech101/256 [8, 12], MSRC [22], PASCAL [7] etc.)已成为当今大多数计算机视觉算法的训练和评估基准。但数据量不足。
- TinyImage是一个拥有8000万张32×32低分辨率图像的数据集，这些数据是通过将WordNet中的所有单词作为查询发送到图像搜索引擎从互联网上收集的。但是高水平的噪声和低分辨率图像使其不太适合通用算法的开发、训练和评估。
- ESP数据集拥有语义层次结构，但整个语义层次结构的分布不够均衡，语义有歧义，导致ESP数据的准确性和实用性可能会受到影响。并且数据集不开源。
- LabelMe 和Lotus Hill数据集分别提供了30k和50k的标记和分段图像，为视觉界提供了补充资源，提供了对象的轮廓和位置。ImageNet缺少此类信息，但类别的数量和每个类别的图像数量已经远远超过了这两个数据集。

#### 5.2 本文是对哪个工作进行的改进

对比其它数据集：

![image-20201212184258425.png](https://upload-images.jianshu.io/upload_images/24435792-d5047727ecdafda6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看出，ImageNet提供了消除歧义的标签(LabelDisam)、干净的注释(clean)、密集的层次结构(DenseHie)、全分辨率图像(FullRes)，可以公开使用(PublicAvail)，大大增加了图像的类别数和图像的数据量，以及图像分辨率。

截至 2016 年，ImageNet 中含有超过 1500 万由人手工注释的图片网址，标签说明了图片中的内容，超过 2.2 万个类别。其中，至少有 100 万张里面提供了边框信息（bounding box）。

#### 5.3 这个领域的关键研究者

李飞飞等人建立的ImageNet 基于 WordNet和Caltech101（2004 年一个专注于图像分类的数据集，也是李飞飞开创的）。ImageNet 不但是计算机视觉发展的重要推动者，也是这一波深度学习热潮的关键驱动力之一。

### 6 论文提出的解决方案的关键

如何构建一个符合数量和精度要求的数据库？

#### 6.1 收集候选图像

利用多个图像搜索引擎从Internet收集图像。对于每个同义词集，查询的是WordNet同义词的集合。例如，在查询“ whippet”时，根据WordNet的表述，“small slender dog of greyhound type developed in England”，换用不同的修饰词“ whippet dog”和“ whippet greyhound”来查询。

同时利用WordNets里的其它语言，比如中文，西班牙语，意大利语等查询。

#### 6.2 清洁候选图像

通过Amazon Mechanical Turk（AMT）服务征集人手，人工标注图像的类别。对每张图片进行AMT用户标记，直到达到预定的置信度得分阈值，以此来过滤候选图像，从而导致每个同义词集内的清晰图像百分比都很高。

![image-20201212193658515.png](https://upload-images.jianshu.io/upload_images/24435792-f93f86e5ae3affab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 7 论文的解决方案有完备的理论证明吗

Torralba等人 [24] 已经证明，给定大量的图像，尽管噪声水平很高，简单的最近邻方法仍可以实现合理的性能。也就是本文实验中用到的分类方法[NN-voting](https://ieeexplore.ieee.org/document/4531741)、[NBNN](http://www.wisdom.weizmann.ac.il/~irani/PAPERS/InDefenceOfNN_CVPR08.pdf)。基于此，扩大数据集容量和精度都是必要的。

### 8 实验设计

#### 8.1用到了哪些数据集

测试的即是ImageNet的性能。测试数据集为Caltech256。

#### 8.2与什么算法进行了比较

##### 对象识别，测试数据集Caltech256

ImageNet使用清晰的全分辨率图像集，对象识别可以更加准确，尤其是通过利用更多的特征级别的信息。

![image-20201212174124543.png](https://upload-images.jianshu.io/upload_images/24435792-ac110c0f01c5f91f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

测试图像为Caltech256，利用哺乳动物子树之间16个常见类别的图像进行测试。

- NN-voting + noisy ImageNet：利用最近邻方法（NN-voting）[24]，给定查询图像，通过与哺乳动物子树的SSD像素距离检索了100个最邻近图像。然后，通过汇总目标类别树内的投票（最近邻居的数目）来执行分类。noisy即无需人工清洁，仅仅从搜索引擎收集的图像。
- NN-voting + clean ImageNet：在干净的ImageNet数据集（有人工清洁）上运行与相同的NN-voting实验。该结果表明，具有更准确的数据可以提高分类性能。
- NBNN：利用朴素贝叶斯最近邻方法（NBNN）[5]，以强调全分辨率图像的有用性。结果表明NBNN可以提供更好的性能，这表明使用通过全分辨率图像提供的更复杂的特征表示的优势。（对NBNN用不同的数据集Caltech256和ImageNet，发现使用全分辨率图像可以提高精度。）
- NBNN-100：运行相同的NBNN实验，但将每个类别的图片数限制为100。结果证实了[24]的发现。通过扩大数据集可以显著提高性能。值得注意的是，NBNN-100在访问整个数据集方面胜过NN-voting，再次证明了通过使用全分辨率图像获得详细的特征级别信息的好处。

##### 图像分类，层次结构 vs 独立结构

与其他可用数据集相比，ImageNet以密集的层次结构提供图像数据。

也就是：不仅要考虑节点（例如“ dog”）的分类，还要考虑其子同义词集（例如“德国牧羊犬”，“英国小猎犬”等）的分类。

在哺乳动物子树上进行实验，使用基于AdaBoost的分类器。对于每个同义词集，随机采样90％的图像以形成正训练图像集，其余10％用作测试图像。实验对比发现，树最大分类器的性能始终高于独立分类器。

![image-20201212183411098.png](https://upload-images.jianshu.io/upload_images/24435792-7ea15540d2c4beef.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 扩展：对象定位算法的基准数据集

作者想到的可能的扩展，并提供了对数据的更多见解：

可以扩展ImageNet能提供的每个图像的其他信息，比如确定每个图像中对象的空间范围。其次可以在混乱的场景中定位对象可以使用户将ImageNet用作对象定位算法的基准数据集。也就是为图像预先定好了边界框：

![image-20201212184127082.png](https://upload-images.jianshu.io/upload_images/24435792-bce6f5a28abfb0fa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 8.3评价指标是什么

证明ImageNet的优点：

通过扩大数据集可以显著提高性能：NBNN比NBNN-100好。

使用通过全分辨率图像提供的更复杂的特征表示的优势：NBNN比NN-voting好。

具有更准确的数据可以提高分类性能：NN-voting + clean ImageNet比NN-voting + noisy ImageNet好。

ImageNet的密集的层次结构的优越性：ImageNet实验中树最大分类器的性能始终高于独立分类器。

#### 8.4有没有什么独特的实验实验设计？

方法都比较传统，本论文主要测试ImageNet的性能。

### 9 实验支撑

#### 9.1 论文的数据集哪里获取

在http://image-net.org/request内可申请下载原始图片，其中要求使用学校的电子邮箱注册账号，之后官方会向相应注册邮箱发送下载链接。

之后便可以转到如图所示的下载页面：

![image.png](https://upload-images.jianshu.io/upload_images/24435792-e7e3944900e1d008.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在搜索框内可以查询想要的同义词集合，并进入如下的层次结构下载索引窗口，进行下载。

![image.png](https://upload-images.jianshu.io/upload_images/24435792-398beb952ebe9f64.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 9.2 ImageNet使用方法

以ImageNet数据集下的小数据集Tiny为例，展示如何利用代码读取ImageNet数据集的图片和对应标签。

下载了tiny数据集并解压，文件中包含了train、val和test原始图像数据，如图所示。

![image.png](https://upload-images.jianshu.io/upload_images/24435792-06e4d88c58b99a48.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- train文件夹存放了训练样本，子文件夹名‘n01443537’等为图片的对应WordNet（词典）ID。每个WordNet ID映射到一个特定的单词/对象。train目录下一共含有200个子文件夹。
- val文件夹存放验证样本，并包含文档‘val_annotations.txt’ 对应每张图片的标签
- words.txt是wordNet ID对应的标签实际名称，如'n00001740	entity'
- wnids.txt是tiny数据库拥有的WordNet ID，一共200个

**代码导入数据集**

初始设置：

```python
import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from sklearn import preprocessing

NUM_CLASSES = 200 #训练集集含有的标签数量
NUM_IMAGES_PER_CLASS = 500 #初设每个标签内有500张图片
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS #训练图片数量初始设定，实际图片数量为98179

NUM_VAL_IMAGES = 10000 #验证集数量

TRAINING_IMAGES_DIR = 'D:/深度学习/tiny-imagenet-200/train/' #设置数据集路径
VAL_IMAGES_DIR = 'D:/深度学习/tiny-imagenet-200/val/'
IMAGE_SIZE = 64 #图片分辨率，64*64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS 
```

读取训练集数据：

```python
def load_training_images(image_dir, batch_size=500):
    """
    读取训练集数据

    Parameters:
      image_dir：地址'D:/深度学习/tiny-imagenet-200/train/'
      batch_size=500：限制每个标签最大索引数量（500张）
        
    Returns:
      images：所有训练集图片转化为数组形式，行-图片总数（此处设定为200*500），列-64*64*3
      np.asarray(labels)：标签集合，如['n01443537' 'n01443537'  'n01443537' ...]'
      np.asarray(names)：对应的文件名称，如['n01443537_0.JPEG' 'n01443537_1.JPEG' 'n01443537_10.JPEG' ...]
     """
    image_index = 0
    
    images = np.ndarray(shape=(NUM_IMAGES, IMAGE_ARR_SIZE))
    names = []
    labels = []                       
    
    print("Loading training images from ", image_dir)
    # Loop through all the types directories
    for type in os.listdir(image_dir):
        if os.path.isdir(image_dir + type + '/images/'):
            type_images = os.listdir(image_dir + type + '/images/')
            # Loop through all the images of a type directory
            batch_index = 0;
            #print ("Loading Class ", type)
            for image in type_images:
                image_file = os.path.join(image_dir, type + '/images/', image)

                # reading the images as they are; no normalization, no color editing
                image_data = mpimg.imread(image_file) 
                #print ('Loaded Image', image_file, image_data.shape)
                if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                    images[image_index, :] = image_data.flatten()

                    labels.append(type)
                    names.append(image)
                    
                    image_index += 1
                    batch_index += 1
                if (batch_index >= batch_size):
                    break;
    
    print("Loaded Training Images", image_index)
    return (images, np.asarray(labels), np.asarray(names))
```

读取验证集数据，根据文档‘val_annotations.txt’ 的标签内容：

```python
def get_label_from_name(data, name):
    """
    根据标签字典‘val_annotations.txt’内对应的标签类别查找图片标签
    
    Parameters:
      data：文档‘val_annotations.txt’ 的标签内容
      name：图片名称    
    Returns:    
      None
    """
    for idx, row in data.iterrows():       
        if (row['File'] == name):
            return row['Class'] #返回对应的标签类别，如'n02823428'
        
    return None

def load_validation_images(testdir, validation_data, batch_size=NUM_VAL_IMAGES):
    """
    读取验证集数据

    Parameters:
      testdir：地址'D:/深度学习/tiny-imagenet-200/val/'
      validation_data：读取到文档‘val_annotations.txt’ 的标签内容，在运行部分有相应的读取代码
      batch_size=NUM_VAL_IMAGES：限制索引数量不得超过设定的验证集数量（10000）
        
    Returns:
      images：所有训练集图片转化为数组形式，行-图片总数（验证集在初始设定时为10000），列-64*64*3
      np.asarray(labels)：标签集合，如['n03444034', 'n04067472' ...]'
      np.asarray(names)：对应的文件名称，如['val_0.JPEG' 'val_1.JPEG'...]
     """    
    labels = []
    names = []
    image_index = 0
    
    images = np.ndarray(shape=(batch_size, IMAGE_ARR_SIZE))
    val_images = os.listdir(testdir + '/images/')
           
    # Loop through all the images of a val directory
    batch_index = 0;
    
    
    for image in val_images:
        image_file = os.path.join(testdir, 'images/', image)
        #print (testdir, image_file)

        # reading the images as they are; no normalization, no color editing
        image_data = mpimg.imread(image_file) 
        if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
            images[image_index, :] = image_data.flatten()
            image_index += 1
            labels.append(get_label_from_name(validation_data, image))
            names.append(image)
            batch_index += 1
            
        if (batch_index >= batch_size):
            break;
    
    print ("Loaded Validation images ", image_index)
    return (images, np.asarray(labels), np.asarray(names))
```

运行：


```python
#读取训练集数据
training_images, training_labels, training_files = load_training_images(TRAINING_IMAGES_DIR)

#读取文档‘val_annotations.txt’ 的标签内容
val_data = pd.read_csv(VAL_IMAGES_DIR + 'val_annotations.txt', sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W']) 
#读取测试集数据
val_images, val_labels, val_files = load_validation_images(VAL_IMAGES_DIR, val_data)
```

### 10 实验结果是否验证了科学假设？

是，应用领域的实验见第8节，可以看出ImageNet的优势。

### 11 论文最大的贡献

"ImageNet改变了AI领域人们对数据集的认识，人们真正开始意识到它在研究中的地位，就像算法一样重要"，李飞飞。

一个好的数据集相当于给算法一个好的试验田。在2010 年开始了 7 届 ImageNet 挑战赛ImageNet Large Scale Visual Recognition Challenge(ILSVRC)。比赛项目包括：图像分类(Classification)、目标定位(Object localization)、目标检测(Object detection)、视频目标检测(Object detection from video)、场景分类(Scene classification)、场景解析(Scene parsing)。ILSVRC中使用到的数据仅是ImageNet数据集中的一部分。

随后，神经网络变得越来越流行，如2012年 AlexNet的论文《imagenet classification with deep convolutional neural networks》，提高了准确率，以及GoogLeNet、VGG Net（2014）、ResNet（2015）。

神经网络流行的同时，网络上的图像数据量有了爆发性的增长，GPU 的性能也在飞速提升，三者合力的结果就是为人类带来了一场席卷全球的深度学习革命。

下面这张图总结 2010-2016 年的 ImageNet 挑战赛成果：分类错误率从 0.28 降到了 0.03；物体识别的平均准确率从 0.23 上升到了 0.66。

![img](http://img.mp.itc.cn/upload/20170727/69b044573c5940b7b01e2a10b4dbe365_th.jpg)

### 12 论文的不足之处

#### 12.1 这篇论文之后的工作有哪些其他的改进

##### 完成ImageNet

论文发表时还仅仅涵盖WordNet同义词集的〜10％，目标是在大约5万个同义词集中散布着大约5000万张清晰、多样化和全分辨率的图像。

扩展ImageNet，使其包含更多信息，例如例子中所述的定位信息、图像的分割、跨同义词集引用，以及针对困难的同义词集的专家标注；之后建立ImageNet社区并开发一个在线平台，每个人都可以为ImageNet资源做出贡献并从中受益。

##### 利用ImageNet

- 培训资源：ImageNet包含几乎所有对象类，也包括稀有类的大量图像。一个有趣的研究方向可能是转移常见对象的知识以学习稀有对象模型。
- 作为基准数据集。
- 为视觉建模引入新的语义关系：因为ImageNet唯一地链接到WordNet的所有具体名词，它们的同义词集之间有着很强的相互联系，所以也可以利用不同的语义关系作为例子学习部分模型。为了全面了解场景，考虑语义层次结构的不同深度也很有帮助。
- 人类视觉研究：ImageNet丰富的结构和对图像世界的密集覆盖可能有助于增进对人类视觉系统的理解。

##### ImageNet 的未来

现如今 ImageNet 已经有 13Million（百万）标注图像。

ImageNet 2017 挑战赛是最后一届，但 image-net.org 仍然会一直存在，并致力于为计算机视觉做出更大的贡献。

自 ImageNet 以来，很多科技巨头都陆续开放了大规模图像数据集。如谷歌在 2016 年发布了 Open Images 数据集，该数据集包含 6000 多个类别共计 9M 图像，还有 JFT-300M 数据集，该数据集有 300M 非精确标注的图像。因此 ImageNet 的未来可能会催生一批大规模开放数据集。

#### 12.2你觉得可以对这篇论文有什么改进

目前 ImageNet 对比其它已有数据集，确少图像分割的注释（segmentation annotations），可以考虑在未来工作对数据集增加图像分割的信息。

### 13 重要的相关论文

1. WordNet数据库：https://wordnet.princeton.edu/

   以及相关论文：[WordNet: An Electronic Lexical Database](https://www.researchgate.net/publication/307972585_WordNet_An_Electronic_Lexical_Database)

2. Torralba等人 [24] 已经证明，给定大量的图像，尽管噪声水平很高，简单的最近邻方法仍可以实现合理的性能。也就是本文实验中用到的分类方法[NN-voting](https://ieeexplore.ieee.org/document/4531741)、[NBNN](http://www.wisdom.weizmann.ac.il/~irani/PAPERS/InDefenceOfNN_CVPR08.pdf)

3. 李飞飞的总结性论文：[ImageNet Large Scale Visual Recognition Challenge](https://link.springer.com/article/10.1007/s11263-015-0816-y) 

   总结了2010-2014年以来ImageNet比赛中关于图像分类和物体识别领域的研究。

### 14 不懂之处

使用全分辨率图像，是不是都用ImageNet？为什么把最近邻方法（NN-voting）换成朴素贝叶斯最近邻方法（NBNN）就可以证明使用全分辨率图像的有效性？此处可能需要细看[5]和[24]的论文。

答：因为NBNN没有和NN-voting对比，而对NBNN用不同的数据集Caltech256和ImageNet，发现使用全分辨率图像可以提高精度。
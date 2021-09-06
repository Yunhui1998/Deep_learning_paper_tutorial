# ILSVRC竞赛介绍（ImageNet Large Scale Visual Recognition Challenge）

## ImageNet

1. 是一个超过15 million的图像数据集，大约有22,000类。
2. 是由**李飞飞**团队从2007年开始，耗费大量人力，通过各种方式（网络抓取，人工标注，亚马逊众包平台）收集制作而成，它作为论文在CVPR-2009发布。当时人们还很怀疑通过更多数据就能改进算法的看法。
3. 深度学习发展起来有几个关键的因素，一个就是庞大的数据（比如说ImageNet），一个是GPU的出现。（还有更优的深度模型，更好的优化算法，可以说数据和GPU推动了这些的产生，这些产生继续推动深度学习的发展）。

## ILSVRC

官方网站：http://www.image-net.org/challenges/LSVRC/

1. 是一个比赛，全称是ImageNet Large-Scale Visual Recognition Challenge，平常说的ImageNet比赛指的是这个比赛。
2. 使用的数据集是ImageNet数据集的一个子集，一般说的ImageNet（数据集）实际上指的是**ImageNet的这个子集**，总共有1000类，每类大约有1000张图像。具体地，有大约1.2 million的训练集，5万验证集，15万测试集。
3. ILSVRC从2010年开始举办，到2017年是最后一届。ILSVRC-2012的数据集被用在2012-2014年的挑战赛中（VGG论文中提到）。ILSVRC-2010是唯一提供了test set的一年。
4. ImageNet可能是指整个数据集（15 million），也可能指比赛用的那个子集（1000类，大约每类1000张），也可能指ILSVRC这个比赛。需要根据语境自行判断。

### 主要竞赛的项目

**（1）图像分类与目标定位（CLS-LOC）**

**图像分类**的任务是要判断图片中物体在1000个分类中所属的类别，主要采用**top-5错误率**的评估方式，即对于每张图给出5次猜测结果，只要5次中有一次命中真实类别就算正确分类，最后统计没有命中的错误率。

2012年之前，图像分类最好的成绩是26%的错误率，2012年AlexNet的出现降低了10个百分点，错误率降到16%。2016年，公安部第三研究所选派的“搜神”（Trimps-Soushen）代表队在这一项目中获得冠军，将成绩提高到仅有2.9%的错误率。

**目标定位**是在分类的基础上，**从图片中标识出目标物体所在的位置**，用**方框**框定，以错误率作为评判标准。目标定位的难度在于图像分类问题可以有5次尝试机会，而在目标定位问题上，每一次都需要框定的非常准确。

目标定位项目在2015年ResNet从上一年的最好成绩25%的错误率提高到了9%。2016年，公安部第三研究所选派的“搜神”（Trimps-Soushen）代表队的错误率仅为7%。

**（2）目标检测（DET）**

目标检测是在定位的基础上更进一步，在**图片中同时检测并定位多个类别的物体**。具体来说，是要在每一张测试图片中找到属于200个类别中的所有物体，如人、勺子、水杯等。**评判方式是看模型在每一个单独类别中的识别准确率**，在多数类别中都获得最高准确率的队伍获胜。**平均检出率mean AP（mean Average Precision）**也是重要指标，一般来说，平均检出率最高的队伍也会多数的独立类别中获胜，2016年这一成绩达到了66.2。

**（3）视频目标检测（VID）**

**视频目标检测**是要检测出**视频每一帧中包含的多个类别的物体**，与图片目标检测任务类似。要检测的目标物体有30个类别，是目标检测200个类别的子集。此项目的最大难度在于要求算法的检测效率非常高。评判方式是在独立类别识别最准确的队伍获胜。

2016年南京信息工程大学队伍在这一项目上获得了冠军，他们提供的两个模型分别在10个类别中胜出，并且达到了平均检出率超过80%的好成绩。

**（4）场景分类（Scene）**

**场景分类**是识别**图片中的场景**，比如森林、剧场、会议室、商店等。也可以说，场景分类要识别图像中的背景。这个项目由MIT Places团队组织，使用**Places2数据集**，包括400个场景的超过1000万张图片。评判标准与图像分类相同（top-5），5次猜测中有一次命中即可，最后统计错误率。

2016年最佳成绩的错误率仅为9%。

场景分类问题中还有一个子问题是**场景分割**，是将**图片划分成不同的区域**，比如天空、道路、人、桌子等。该项目由MIT CSAIL视觉组织，使用**ADE20K数据集**，包含2万张图片，150个标注类别，如天空、玻璃、人、车、床等。这个项目会同时评估像素及准确率和分类IOU（Intersection of Union）

### 历届冠军

![img](https://img-blog.csdnimg.cn/20181218112845274.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTc3MDE2OQ==,size_16,color_FFFFFF,t_70)

**ImageNet的分类结果（加粗为冠军）**

|  年  |     网络/队名      | val top-1 | val top-5 | test top-5 |           备注            |
| :--: | :----------------: | :-------: | :-------: | :--------: | :-----------------------: |
| 2012 |      AlexNet       |   38.1%   |   16.4%   |   16.42%   |          5 CNNs           |
| 2012 |    **AlexNet**     |   36.7%   |   15.4%   |   15.32%   |  7CNNs。用了2011年的数据  |
| 2013 |      OverFeat      |           |           |   14.18%   |       7 fast models       |
| 2013 |      OverFeat      |           |           |   13.6%    |    赛后。7 big models     |
| 2013 |       ZFNet        |           |           |   13.51%   |  ZFNet论文上的结果是14.8  |
| 2013 |      Clarifai      |           |           |   11.74%   |                           |
| 2013 |    **Clarifai**    |           |           |   11.20%   |     用了2011年的数据      |
| 2014 |        VGG         |           |           |   7.32%    |    7 nets, dense eval     |
| 2014 |    VGG（亚军）     |   23.7%   |   6.8%    |    6.8%    |       赛后。2 nets        |
| 2014 |  **GoogleNet v1**  |           |           |   6.67%    |     7 nets, 144 crops     |
|      |    GoogleNet v2    |   20.1%   |   4.9%    |   4.82%    |  赛后。6 nets, 144 crops  |
|      |    GoogleNet v3    |   17.2%   |   3.58%   |            |  赛后。4 nets, 144 crops  |
|      |    GoogleNet v4    |   16.5%   |   3.1%    |   3.08%    | 赛后。v4+Inception-Res-v2 |
| 2015 |     **ResNet**     |           |           |   3.57%    |         6 models          |
| 2016 | **Trimps-Soushen** |           |           |   2.99%    |         公安三所          |
| 2016 |  ResNeXt（亚军）   |           |           |   3.03%    |   加州大学圣地亚哥分校    |
| 2017 |     **SENet**      |           |           |   2.25%    |    Momenta 与牛津大学     |

**ImageNet的定位结果（加粗为冠军）**

|  年  |     网络/队名      | val top-5 | test top-5 |                    备注                    |
| :--: | :----------------: | :-------: | :--------: | :----------------------------------------: |
| 2012 |      AlexNet       |           |   34.19%   |          多伦多大学Hinton和他学生          |
| 2012 |    **AlexNet**     |           |   33.55%   |              用了2011年的数据              |
| 2013 |    **OverFeat**    |   30.0%   |   29.87%   |             纽约大学Lecun团队              |
| 2014 |     GoogleNet      |           |   26.44%   |                    谷歌                    |
| 2014 |      **VGG**       |   26.9%   |   25.32%   |                  牛津大学                  |
| 2015 |     **ResNet**     |   8.9%    |   9.02%    |                    微软                    |
| 2016 | **Trimps-Soushen** |           |   7.71%    | 公安三所，以Inception, resNet, WRN等为基础 |
| 2017 |      **DPN**       |           |   6.23%    |          新加坡国立大学与奇虎360           |

**ImageNet的检测结果（加粗为冠军）**

|  年  |   网络/队名   | mAP(%) |               备注               |
| :--: | :-----------: | :----: | :------------------------------: |
| 2013 |   OverFeat    | 19.40  |    使用了12年的分类数据预训练    |
| 2013 |    **UvA**    | 22.58  |                                  |
| 2013 |   OverFeat    |  24.3  | 赛后。使用了12年的分类数据预训练 |
| 2014 | **GoogleNet** | 43.93  |              R-CNN               |
| 2015 |  **ResNet**   | 62.07  |           Faster R-CNN           |
| 2016 |  **CUImage**  | 66.28  | 商汤和港中文，以GBD-Net等为基础  |
| 2017 |   **BDAT**    | 73.41  |  南京信息工程大学和帝国理工学院  |

**其它**
HikVision（海康威视）：2016年的场景分类第一



### 现状

由于深度学习技术的日益发展，使得机器视觉在ILSVRC的比赛成绩屡创佳绩，其错误率已经低于人类视觉，若再继续举办类似比赛已无意义，是故大家对电脑视觉技术的期待由相当成熟的 image identification 转向尚待开发的 image understanding 。

ILSVRC 2017 已是最后一届举办。2018年起，将由WebVision竞赛（Challenge on Visual Understanding by Learning from Web Data）来接棒。WebVision所使用的dataset抓取自浩瀚的网络，不经过人工处理与label，难度大大提高，但也会更加贴近实际运用场景。

正是因为ILSVRC 2012挑战赛上的AlexNet横空出世，使得全球范围内掀起了一波深度学习热潮。这一年也被称作“深度学习元年”。此后，ILSVRC挑战赛的名次一直是衡量一个研究机构或企业技术水平的重要标尺。因此，即使ILSVRC挑战赛停办了，但其对深度学习的深远影响和巨大贡献，将永载史册。





参考博客：

[1]: https://www.cnblogs.com/liaohuiqiang/p/9609162.html	"ImageNet历届冠军和相关CNN模型"
[2]: https://blog.csdn.net/Sophia_11/article/details/84570177?utm_medium=distribute.pc_relevant.none-task-blog-searchFromBaidu-3.control&amp;depth_1-utm_source=distribute.pc_relevant.none-task-blog-searchFromBaidu-3.control	"ILSVRC竞赛详细介绍（ImageNet Large Scale Visual Recognition Challenge）"
[3]: https://blog.csdn.net/weixin_41770169/article/details/85062229	"深度学习: ILSVRC竞赛（ImageNet竞赛）"


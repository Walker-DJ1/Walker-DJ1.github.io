---
layout: post
comments: true
published: True
title: "Hierarchical Text Classification:层次文本分类"
excerpt: "文本分类 (Text Classification, TC)是自然语言处理 (NLP) 社区中研究最广泛的任务。分层文本分类（Hierarchical Text Classification , HTC）是 TC 的子任务，也是更广泛的分层多标签分类 (HMC) 的一部分。Hierarchical Text Classification 与普通的Text Classification任务不同，预测标签之间存在分级关系，预测得到的标签集合需要满足预先设定好的类目树关系"
date:   2024-02-01 07:00:00
mathjax: false
---

## 关键词

- hierarchical textclassification
- hierarchical multilabel classification

* * *

## 引言

**文本分类 (Text Classification, TC)** 是自然语言处理 (NLP) 社区中研究最广泛的任务。分层文本分类（Hierarchical Text Classification , HTC）是 TC 的子任务，也是更广泛的分层多标签分类 (HMC) 的一部分。==Hierarchical Text Classification 与普通的Text Classification任务不同，预测标签之间存在分级关系，预测得到的标签集合需要满足预先设定好的类目树关系==

- 常见的文本分类任务中类目之间通常是正交的，即不存在包含关系，标签之间是扁平化的。
- 层次分类则是一类特殊的文本分类任务，即类目之间存在层次结构关系，一般可以表示为树形或者无向图。它们的标签以多级层次结构组织，其中每个标签可以被视为能够拥有许多可能的子节点的节点
- 在这类任务中，一条样本的标签会同时包括层次结构中的父类和子类目，在这些类型的情况下，层次结构是实现良好分类方法的关键。更复杂的情况下，一条样本会同时包含层次结构中多条路径，这类任务则称为层次多标签任务（HMC，Hierarchical Multi-Classification）
- 近年来提出了大量 HTC 新方法。这些“分层分类器”的优势在于它们**能够利用标签之间的依赖关系来提高其分类性能**。

## 层次分类的类型

### 层次结构分类

标准分类器关注 HTC 定义的平面分类。其中，类别被单独对待（即，彼此之间没有关系）。相比之下，HTC 处理文档的标签组织结构类似于树或有向无环图 (DAG),在这些结构中，每个节点都包含要分配的标签，如图所示。  
![分层结构的标签](https://github.com/Walker-DJ1/Walker-DJ1.github.io/blob/master/image_data/01Hierarchical%20Text%20Classification/fig1.png?raw=true)

HTC 方法可以分为两组：**局部方法和全局方法**。局部方法（有时称为“自上而下”）被如此定义，因为它们“剖析”层次结构，构造多个与节点标签子集一起使用的局部分类器。虽然比平面分类器（忽略层次结构）信息更丰富，但不可避免地会丢失层次信息，因为这些分类器的聚合往往会忽略分类层次结构的整体结构信息。

- **局部方法:** 通过学习不同层类目和文本之间的关系，并聚合不同层的预测结果来得到最终的预测结果。这类方法通常由多个分类模块组成，例如自顶向下的层次分类，在每个非叶子节点都有一个局部分类器，在预测时根据父类目的预测结果来预测最终的子类目，因其结构问题而受到批评，最值得注意的一点是它们可能很容易传播错误分类。此外，这些模型的可训练参数通常很大，并且由于缺乏整体结构信息，很容易面临暴露偏差。
- **全局方法:** 旨在解决局部方法这些缺点，由一个分类模块构成，直接利用层级结构信息来建模。例如利用层次结构来构造递归正则化损失项来对分类参数进行约束。虽然全局分类器的定义是故意通用的，但人们可以将直接考虑层次结构的任何分类算法想象为全局。还值得注意的是，全球和本地方法也可以结合起来。

### 最常见的 HTC 方法

以树状层次结构为例。扁平化分类器（a）丢失所有层次信息，而局部分类器（b、c、d）可以合并其中一些信息。全局分类器（e）旨在充分利用标签结构。

![分层结构的标签](https://github.com/Walker-DJ1/Walker-DJ1.github.io/blob/master/image_data/01Hierarchical%20Text%20Classification/fig2.png?raw=true)

- ==扁平化分类器 (Flattened classifiers)==:扁平化分类方法将任务简化为多类（或多标签）分类问题，因此完全丢弃层次信息。通常，仅考虑叶节点，并且层次结构的更高级别的分类继承自父节点
- ==局部分类器(Local classifiers)==:局部分类方法通常分为三类，这三类在剖析层次结构的方式上有所不同。
    - local per-node (binary) approaches:本地每节点（二进制）方法的类（图 b）将每个标签视为一个单独的类，完全忽略层次结构。
    - Local per-level methods:本地每级别方法（图 c）为每个相关类别级别分配一个分类器，该分类器的任务是单独为该级别做出决定。
    - local per-parent methods:本地每父节点方法（图 d）将一个分类器分配给所有父节点，并要求它们将样本分配给其子节点之一，从而捕获样本当前通过层次结构的部分路径。
- ==全局分类方法(global classifiers)==: 利用整个层次结构来做出最终的分类决策。全局分类器在扁平表示上执行实际分类是很常见的（尽管不是绝对必要的）；因此，层次结构信息是通过结构偏差（即模型的架构）来实现的.

### 层次文本分类方法

概述了为解决 2019-2022 年期间 HTC 任务而设计的方法

- HTrans 在他们的工作中，Banerjee 等人。 (2019) 提出了分层迁移学习 (HTrans)，这是一种提高本地每节点分类方法性能的框架。一般的直觉是，可以通过使用其父分类器的参数初始化较低级别的分类器来将知识传递给较低级别​​的分类器。首先，他们利用带有注意力机制增强的基于 GRU 的双向 RNN 作为文本编码器，然后使用全连接网络作为解码器来生成类别概率。词嵌入使用 GloVe 预训练嵌入进行初始化。使用二进制输出为层次结构树中的每个节点训练一个这样的模型，并且子节点与祖先节点的分类器共享参数。这可以被视为一种“硬”共享方法，它利用微调来强制从父节点到子节点的归纳偏差。推理是通过标准的自上而下的方法实现的。作者通过删除参数共享和注意力来进行消融研究，并将结果与​​使用二元分类器的权重初始化的多标签模型进行比较，在包括他们提出的增强功能时展示了坚实的改进。
- **HiAGM** 提出了一种端到端层次结构感知全局模型（HiAGM），该模型利用细粒度层次结构信息并聚合标签式文本特征。直观上，他们的目标是通过引入层次结构感知结构编码器（结构就是层次结构）来向传统文本编码器添加信息。作为结构编码器，作者测试了适应分层结构的 TreeLSTM 和 GCN。此外，他们建议两种不同的框架：一种基于多标签注意力（HiAGM-LA），一种基于文本特征传播（HiAGM-TP）。 HiAGM-LA 利用注意力机制以双向、分层的方式增强标签表示，利用节点输出作为层次感知的标签表示。另一方面，HiAGM-TP 基于串行数据流中的文本特征传播；文本特征用作结构编码器的直接输入，在整个层次结构中传播信息。对于多标签分类，使用二元交叉熵 (BCE) 损失以及 MATCH 中描述的正则化项 𝑅 参数。

### 结论

- 总体而言，==基于 Transformer 的模型在性能和易用性方面仍然表现出色==。这种迁移学习技术很容易适应新的数据集，并提供全面的出色结果。它的缺点来自于计算成本；训练和微调这些模型是迄今为止最昂贵的过程，而且推理时间也是所有方法中最慢的。因此，它们在实时系统上的使用可能会出现问题，具体取决于要同时处理的数据量。传统的、基于 SVM 的简单应用方法可能是一个很好的选择，尽管也可以设计出更精细的分层方法。
- ==HiAGM 是我们分析的许多作品的最先进的参考==，效果非常好，并为注入分层信息以改进分类提供了支持。在可用性方面，它的适应相对简单，尽管必须进行一些更改才能适用于与原始作品中设计的不同的层次结构。

### 开源项目

| 项目名称 | Description |
| --- | --- |
| 腾讯多层分类项目 |https://github.com/Tencent/NeuralNLP-NeuralClassifier |
| 阿里NLP（best） |https://github.com/Alibaba-NLP/HiAGM |
| Hierarchical-Multi-Label-Text-Classification | https://github.com/RandolphVI/Hierarchical-Multi-Label-Text-Classification|
| [AAAI 2019] 弱监督分层文本分类 (github.com) |https://github.com/yumeng5/WeSHClass|
| HiLAP：论文“分层文本分类与强化标签分配”EMNLP 2019 | https://github.com/morningmoni/HiLAP|
| 用于基于决策树的分层多分类的 Python 模块 | https://github.com/davidwarshaw/hmc |


* * *

> Hierarchical Text Classification: a review of current research  
> https://zhuanlan.zhihu.com/p/409151723  
> https://zhuanlan.zhihu.com/p/76368437  
> https://zhuanlan.zhihu.com/p/152235686

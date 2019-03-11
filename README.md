# PyTorch深度学习：60分钟入门与实战

> [Deep Learning with PyTorch A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html#deep-learning-with-pytorch-a-60-minute-blitz) 中文翻译版，翻译不对的地方拜托大家指出~

>**对PyTorch感兴趣的童鞋欢迎看这个**-->[PyTorch教程、例子和书籍合集](https://github.com/bat67/pytorch-tutorials-examples-and-books)


## 目录

- [PyTorch深度学习：60分钟入门与实战](#pytorch深度学习60分钟入门与实战)
  - [目录](#目录)
  - [1、简介](#1简介)
  - [2、环境](#2环境)
  - [3、目录](#3目录)
    - [3.1、什么是PyTorch？（What is PyTorch?）](#31什么是pytorchwhat-is-pytorch)
    - [3.2、Autograd：自动求导](#32autograd自动求导)
    - [3.3、神经网络（Neural Networks）](#33神经网络neural-networks)
    - [3.4、训练分类器（Training a Classifier）](#34训练分类器training-a-classifier)
    - [3.5 选读：数据并行处理（Optional: Data Parallelism）](#35-选读数据并行处理optional-data-parallelism)
  - [4、类似项目](#4类似项目)
  - [5、版权信息](#5版权信息)



## 1、简介

此教程的目标：

* 更高层次地理解Pythrch的Tensor库以及神经网络。
* 训练一个小的神经网络模型用于分类图像。

本教程假设读者对`numpy`有基本的了解


## 2、环境

* PyTorch版本0.4及以上（[PyTorch 1.0 **稳定版**](https://pytorch.org/get-started/locally/)已经发布，还有什么理由不更新呢~）
* [torchvision](https://github.com/pytorch/vision) 0.2.1

## 3、目录

### 3.1、什么是PyTorch？（What is PyTorch?）

  * [入门](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#%E5%85%A5%E9%97%A8)
    * [张量](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#%E5%BC%A0%E9%87%8F)
    * [运算](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#%E8%BF%90%E7%AE%97)
  * [NumPy桥](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#numpy%E6%A1%A5)
    * [将torch的Tensor转化为NumPy数组](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#%E5%B0%86torch%E7%9A%84tensor%E8%BD%AC%E5%8C%96%E4%B8%BAnumpy%E6%95%B0%E7%BB%84)
    * [将NumPy数组转化为Torch张量](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#%E5%B0%86numpy%E6%95%B0%E7%BB%84%E8%BD%AC%E5%8C%96%E4%B8%BAtorch%E5%BC%A0%E9%87%8F)
  * [CUDA上的张量](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#cuda上的张量)

### 3.2、Autograd：自动求导

  * [张量](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Autograd_Automatic_Differentiation/Autograd%EF%BC%9A自动求导.md#%E5%BC%A0%E9%87%8F)
  * [梯度](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Autograd_Automatic_Differentiation/Autograd%EF%BC%9A自动求导.md#%E6%A2%AF%E5%BA%A6)


### 3.3、神经网络（Neural Networks）

  * [定义网络](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Neural_Networks/神经网络.md#定义网络)
  * [损失函数](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Neural_Networks/神经网络.md#损失函数)
  * [反向传播](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Neural_Networks/神经网络.md#反向传播)
  * [更新权重](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Neural_Networks/神经网络.md#更新权重)

### 3.4、训练分类器（Training a Classifier）

  * [数据呢？](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Training_a_Classifier/训练分类器.md#数据呢)
  * [训练一个图片分类器](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Training_a_Classifier/训练分类器.md#训练一个图片分类器)
    * [1.加载并标准化CIFAR10](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Training_a_Classifier/训练分类器.md#1加载并标准化cifar10)
    * [2.定义卷积神经网络](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Training_a_Classifier/训练分类器.md#2定义卷积神经网络)
    * [3.定义损失函数和优化器](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Training_a_Classifier/训练分类器.md#3定义损失函数和优化器)
    * [4.训练网络](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Training_a_Classifier/训练分类器.md#4训练网络)
    * [5.使用测试数据测试网络](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Training_a_Classifier/训练分类器.md#5使用测试数据测试网络)
  * [在GPU上训练](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Training_a_Classifier/训练分类器.md#在gpu上训练)
  * [在多GPU上训练](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Training_a_Classifier/训练分类器.md#在多gpu上训练)
  * [接下来要做什么？](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Training_a_Classifier/训练分类器.md#接下来要做什么)

### 3.5 选读：数据并行处理（Optional: Data Parallelism）

  * [导入和参数](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Optional_Data_Parallelism/数据并行处理.md#导入和参数)
  * [虚拟数据集](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Optional_Data_Parallelism/数据并行处理.md#虚拟数据集)
  * [简单模型](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Optional_Data_Parallelism/数据并行处理.md#简单模型)
  * [创建一个模型和数据并行](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Optional_Data_Parallelism/数据并行处理.md#创建一个模型和数据并行)
  * [运行模型](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Optional_Data_Parallelism/数据并行处理.md#运行模型)
  * [结果](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Optional_Data_Parallelism/数据并行处理.md#结果)
    * [2个GPU](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Optional_Data_Parallelism/数据并行处理.md#2个gpu)
    * [3个GPU](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Optional_Data_Parallelism/数据并行处理.md#3个gpu)
    * [8个GPU](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Optional_Data_Parallelism/数据并行处理.md#8个gpu)
  * [总结](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Optional_Data_Parallelism/数据并行处理.md#总结)


## 4、类似项目

* [用例子学习 PyTorch](https://github.com/bat67/pytorch-examples-cn)
* [PyTorch 教程、例子和书籍](https://github.com/bat67/pytorch-tutorials-examples-and-books)
* [PyTorch 深度学习：60分钟入门与实战](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn)


## 5、版权信息

除非额外说明，本仓库的所有公开文档均遵循[署名-非商业性使用-相同方式共享 4.0 国际 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh)许可协议。

您可以自由地：

* 共享 — 在任何媒介以任何形式复制、发行本作品
* 演绎 — 修改、转换或以本作品为基础进行创作

惟须遵守下列条件：

* 署名 — 您必须给出适当的署名，提供指向本许可协议的链接，同时标明是否（对原始作品）作了修改。您可以用任何合理的方式来署名，但是不得以任何方式暗示许可人为您或您的使用背书。
* 非商业性使用 — 您不得将本作品用于商业目的。
* 相同方式共享 — 如果您再混合、转换或者基于本作品进行创作，您必须基于与原先许可协议相同的许可协议 分发您贡献的作品。
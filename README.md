# PyTorch深度学习：60分钟入门与实战

> [Deep Learning with PyTorch A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html#deep-learning-with-pytorch-a-60-minute-blitz) 中文翻译版，翻译不对的地方拜托大家指出~

## 简介

此教程的目标：

* 更高层次地理解Pythrch的Tensor库以及神经网络。
* 训练一个小的神经网络模型用于分类图像。

本教程假设读者对`numpy`有基本的了解


## 环境

* PyTorch版本0.4及以上（[PyTorch 1.0 **稳定版**](https://pytorch.org/get-started/locally/)已经发布，还有什么理由不更新呢~）
* [torchvision](https://github.com/pytorch/vision) 0.2.1

## 目录

### [什么是PyTorch？（What is PyTorch?）](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#%E4%BB%80%E4%B9%88%E6%98%AFpytorch)

  * [入门](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#%E5%85%A5%E9%97%A8)
    * [张量](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#%E5%BC%A0%E9%87%8F)
    * [运算](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#%E8%BF%90%E7%AE%97)
  * [NumPy桥](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#numpy%E6%A1%A5)
    * [将torch的Tensor转化为NumPy数组](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#%E5%B0%86torch%E7%9A%84tensor%E8%BD%AC%E5%8C%96%E4%B8%BAnumpy%E6%95%B0%E7%BB%84)
    * [将NumPy数组转化为Torch张量](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#%E5%B0%86numpy%E6%95%B0%E7%BB%84%E8%BD%AC%E5%8C%96%E4%B8%BAtorch%E5%BC%A0%E9%87%8F)
  * [CUDA上的张量](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/What_is_PyTorch/什么是PyTorch.md#cuda上的张量)

### [Autograd：自动求导（Autograd: Automatic Differentiation）](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Autograd_Automatic_Differentiation/Autograd%EF%BC%9A自动求导.md#autograd%E8%87%AA%E5%8A%A8%E6%B1%82%E5%AF%BC)

  * [张量](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Autograd_Automatic_Differentiation/Autograd%EF%BC%9A自动求导.md#%E5%BC%A0%E9%87%8F)
  * [梯度](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Autograd_Automatic_Differentiation/Autograd%EF%BC%9A自动求导.md#%E6%A2%AF%E5%BA%A6)


### [神经网络（Neural Networks）](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Neural_Networks/神经网络.md#%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)

  * [定义网络](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Neural_Networks/神经网络.md#定义网络)
  * [损失函数](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Neural_Networks/神经网络.md#损失函数)
  * [反向传播](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Neural_Networks/神经网络.md#反向传播)
  * [更新权重](https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn/blob/master/Neural_Networks/神经网络.md#更新权重)

### 训练分类器（Training a Classifier）

### 数据并行处理（Optional: Data Parallelism）


## 版权信息

除非额外说明，本仓库的所有公开文档均遵循 [署名-非商业性使用-相同方式共享 3.0 中国大陆 (CC BY-NC-SA 3.0 CN)](https://creativecommons.org/licenses/by-nc-sa/3.0/cn/) 许可协议。任何人都可以自由地分享、修改本作品，但必须遵循如下条件：

* 署名：必须提到原作者，提供指向此许可协议的链接，表明是否有做修改
* 非商业性使用：不能对本作品进行任何形式的商业性使用
* 相同方式共享：若对本作品进行了修改，必须以相同的许可协议共享
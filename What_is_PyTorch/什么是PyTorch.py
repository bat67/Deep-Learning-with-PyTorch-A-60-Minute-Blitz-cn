#%% [markdown]
# >完整项目地址：https://github.com/bat67/Deep-Learning-with-PyTorch-A-60-Minute-Blitz-cn ，
# 嘤嘤嘤求star~，最新版也会首先更新在github上
# 有误的地方拜托大家指出~
#%% [markdown]
# # 什么是PyTorch？
#%% [markdown]
# PyTorch是一个基于python的科学计算包，主要针对两类人群：
# 
# * 作为NumPy的替代品，可以利用GPU的性能进行计算
# * 作为一个高灵活性、速度快的深度学习平台
#%% [markdown]
# ## 1 入门
#%% [markdown]
# ### 1.1 张量
#%% [markdown]
# `Tensor`（张量）类似于`NumPy`的`ndarray`，但还可以在GPU上使用来加速计算。

#%%
from __future__ import print_function
import torch

#%% [markdown]
# 创建一个没有初始化的5*3矩阵：

#%%
x = torch.empty(5, 3)
print(x)

#%% [markdown]
# 创建一个随机初始化矩阵：

#%%
x = torch.rand(5, 3)
print(x)

#%% [markdown]
# 构造一个填满0且数据类型为`long`的矩阵:

#%%
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#%% [markdown]
# 直接从数据构造张量：

#%%
x = torch.tensor([5.5, 3])
print(x)

#%% [markdown]
# 或者根据已有的tensor建立新的tensor。除非用户提供新的值，否则这些方法将重用输入张量的属性，例如dtype等：

#%%
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

#%% [markdown]
# 获取它的形状：

#%%
print(x.size())

#%% [markdown]
# > **注意**：
# >
# > `torch.Size`本质上还是`tuple`，所以支持tuple的一切操作。
#%% [markdown]
# ### 1.2 运算
#%% [markdown]
# 一种运算有多种语法。在下面的示例中，我们将研究加法运算。
# 
# 加法：形式一

#%%
y = torch.rand(5, 3)
print(x + y)

#%% [markdown]
# 加法：形式二

#%%
print(torch.add(x, y))

#%% [markdown]
# 加法：给定一个输出张量作为参数

#%%
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

#%% [markdown]
# 加法：原位/原地操作（in-place）

#%%
# adds x to y
y.add_(x)
print(y)

#%% [markdown]
# >注意：
# >
# >任何一个in-place改变张量的操作后面都固定一个`_`。例如`x.copy_(y)`、`x.t_()`将更改x
# 
#%% [markdown]
# 也可以使用像标准的NumPy一样的各种索引操作：

#%%
print(x[:, 1])

#%% [markdown]
# 改变形状：如果想改变形状，可以使用`torch.view`

#%%
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

#%% [markdown]
# 如果是仅包含一个元素的tensor，可以使用`.item()`来得到对应的python数值

#%%
x = torch.randn(1)
print(x)
print(x.item())

#%% [markdown]
# >后续阅读：
# >
# >超过100中tensor的运算操作，包括转置，索引，切片，数学运算，
# 线性代数，随机数等，具体访问[这里](https://pytorch.org/docs/stable/torch.html)
#%% [markdown]
# ## 2 NumPy桥
#%% [markdown]
# 将一个Torch张量转换为一个NumPy数组是轻而易举的事情，反之亦然。
# 
# Torch张量和NumPy数组将共享它们的底层内存位置，更改一个将更改另一个。
# 
#%% [markdown]
# ### 2.1 将torch的Tensor转化为NumPy数组

#%%
a = torch.ones(5)
print(a)


#%%
b = a.numpy()
print(b)

#%% [markdown]
# 看NumPy数组是如何改变里面的值的：
#%% [markdown]
# a.add_(1)
# print(a)
# print(b)
#%% [markdown]
# ### 2.2 将NumPy数组转化为Torch张量
#%% [markdown]
# 看改变NumPy数组是如何自动改变Torch张量的：

#%%
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

#%% [markdown]
# CPU上的所有张量(CharTensor除外)都支持转换为NumPy以及由NumPy转换回来。
#%% [markdown]
# ## 3 CUDA上的张量
#%% [markdown]
# 张量可以使用`.to`方法移动到任何设备（device）上：

#%%
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!


#%%




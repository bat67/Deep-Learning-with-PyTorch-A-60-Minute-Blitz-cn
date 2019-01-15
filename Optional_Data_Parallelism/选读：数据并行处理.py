#%% [markdown]
# # 选读：数据并行处理
#%% [markdown]
# 在这个教程里，我们将学习如何使用数据并行（`DataParallel`）来使用多GPU。
# 
# PyTorch非常容易的就可以使用GPU，可以用如下方式把一个模型放到GPU上：

#%%
device = torch.device("cuda：0")
model.to(device)

#%% [markdown]
# 然后可以复制所有的张量到GPU上：

#%%
mytensor = my_tensor.to(device)

#%% [markdown]
# `my_tensor.to(device)`返回一个GPU上的`my_tensor`副本，而不是重写`my_tensor`。我们需要把它赋值给一个新的张量并在GPU上使用这个张量。
# 
# 在多GPU上执行前向和反向传播是自然而然的事。然而，PyTorch默认将只是用一个GPU。你可以使用`DataParallel`让模型并行运行来轻易的让你的操作在多个GPU上运行。

#%%
model = nn.DataParallel(model)

#%% [markdown]
# 这是这篇教程背后的核心，我们接下来将更详细的介绍它。
#%% [markdown]
# ## 导入和参数
#%% [markdown]
# 导入PyTorch模块和定义参数。

#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

#%% [markdown]
# ## 虚拟数据集
#%% [markdown]
# 要制作一个虚拟（随机）数据集，只需实现`__getitem__`。

#%%
class RandomDataset(Dataset)：

    def __init__(self, size, length)：
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index)：
        return self.data[index]

    def __len__(self)：
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

#%% [markdown]
# ## 简单模型
#%% [markdown]
# 作为演示，我们的模型只接受一个输入，执行一个线性操作，然后得到结果。然而，你能在任何模型（CNN，RNN，Capsule Net等）上使用`DataParallel`。
# 
# 我们在模型内部放置了一条打印语句来检测输入和输出向量的大小。请注意批等级为0时打印的内容。

#%%
class Model(nn.Module)：
    # Our model

    def __init__(self, input_size, output_size)：
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input)：
        output = self.fc(input)
        print("\tIn Model： input size", input.size(),
              "output size", output.size())

        return output

#%% [markdown]
# ## 创建一个模型和数据并行
#%% [markdown]
# 这是本教程的核心部分。首先，我们需要创建一个模型实例和检测我们是否有多个GPU。如果我们有多个GPU，我们使用`nn.DataParallel`来包装我们的模型。然后通过`model.to(device)`把模型放到GPU上。

#%%
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1：
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

#%% [markdown]
# ## 运行模型
#%% [markdown]
# 现在我们可以看输入和输出张量的大小。

#%%
for data in rand_loader：
    input = data.to(device)
    output = model(input)
    print("Outside： input size", input.size(),
          "output_size", output.size())

#%% [markdown]
# ## 结果
#%% [markdown]
# 当我们对30个输入和输出进行批处理时，我们和期望的一样得到30个输入和30个输出，但是若有多个GPU，会得到如下的结果。
#%% [markdown]
# ### 2个GPU
#%% [markdown]
# 若有2个GPU，将看到
#%% [markdown]
# ```python
# # on 2 GPUs
# Let's use 2 GPUs!
#     In Model： input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     In Model： input size torch.Size([15, 5]) output size torch.Size([15, 2])
# Outside： input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#     In Model： input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     In Model： input size torch.Size([15, 5]) output size torch.Size([15, 2])
# Outside： input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#     In Model： input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     In Model： input size torch.Size([15, 5]) output size torch.Size([15, 2])
# Outside： input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#     In Model： input size torch.Size([5, 5]) output size torch.Size([5, 2])
#     In Model： input size torch.Size([5, 5]) output size torch.Size([5, 2])
# Outside： input size torch.Size([10, 5]) output_size torch.Size([10, 2])
# ```
#%% [markdown]
# ### 3个GPU
#%% [markdown]
# 若有3个GPU，将看到：
#%% [markdown]
# ```python
# Let's use 3 GPUs!
#     In Model： input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     In Model： input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     In Model： input size torch.Size([10, 5]) output size torch.Size([10, 2])
# Outside： input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#     In Model： input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     In Model： input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     In Model： input size torch.Size([10, 5]) output size torch.Size([10, 2])
# Outside： input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#     In Model： input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     In Model： input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     In Model： input size torch.Size([10, 5]) output size torch.Size([10, 2])
# Outside： input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([2, 5]) output size torch.Size([2, 2])
# Outside： input size torch.Size([10, 5]) output_size torch.Size([10, 2])
# ```
#%% [markdown]
# ### 8个GPU
#%% [markdown]
# 若有8个GPU，将看到：
#%% [markdown]
# ```python
# Let's use 8 GPUs!
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
# Outside： input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
# Outside： input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     In Model： input size torch.Size([2, 5]) output size torch.Size([2, 2])
# Outside： input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#     In Model： input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     In Model： input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     In Model： input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     In Model： input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     In Model： input size torch.Size([2, 5]) output size torch.Size([2, 2])
# Outside： input size torch.Size([10, 5]) output_size torch.Size([10, 2])
# ```
#%% [markdown]
# ## 总结
#%% [markdown]
# `DataParallel`自动的划分数据，并将作业发送到多个GPU上的多个模型。在每个模型完成作业后，`DataParallel`收集并合并结果返回给你。
# 
# 更多信息，请参考：http://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

#%%




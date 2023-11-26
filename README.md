# Federated Learning Project

## 提前标注

**options.py和baseline_main.py, update.py在本次实验中并没有用到，
仅只是本人编写代码时进行参照的内容以及可能之后会用到的代码，
可以忽略**

## utils.py

1. 参数类`parameter`，可以用于传递超参数以及超参数集中化

2. 学习率计算类`eta_class`，定义了三种计算每一轮学习率的计算方式
   
   <img src="file:///Users/michael/Library/Application%20Support/marktext/images/2023-10-29-15-34-07-image.png" title="" alt="" width="316">
   
   // 在这里我们主要适用第二种计算方式

3. 数据导出类`excel_solver`： 输入文件名，`save_excel`负责输出算法的迭代过程，而`save_best_param`负责输出不同算法的最优参数的情况

4. mkdir函数以及make_dir函数，用于创建存储csv文件的文件夹，保证后续的实现

## algorithm.py

定义算法类，目前存储了FedAvg算法和Zeroth-grad算法

算法类的存储有固定格式，每个算法类都会固定分成四个函数

1. `__init__` 用于超参数和模型，数据集的存储，每一次的实验都务必要对模型进行重置

2. `update_weight` 用于模拟单个客户端的更新

3. `average` 用于聚合

4. `alg_run`  负责整个算法的运行

2023.10.16补充： 既然我们的算法本身是零阶的，那么我们选择算法的时候也应该选择零阶的算法，比如说在这里我们需要将常规的FedAvg算法中求梯度的部分改成使用有限微分法求梯度的零阶算法

## test.py

牛顿法验证代码，牛顿法精度较高，通过对比牛顿法的损失函数值和我们写的损失函数值即可判断代码的准确性

如果把超参数的值调高，在评估次数为3000*数据集大小的时候，两个数据集使用零阶优化的算法计算的损失函数与牛顿法的相差不大于0.03，所以可以认为如果算法能够跑起来，得到的损失函数结果应该是没有问题的

## models.py

存储了模型信息，在这里我们目前只使用了逻辑回归这一模型，在条件允许的情况下，可以使用多个模型来辅助实验

模型主要完成求梯度`def grad()`和计算损失函数`def loss()`两个功能

2023.10.16补充：

1. 根据老师建议，无梯度的算法就直接包含在算法类的`update_client`函数中，没有在这里封装
2. 需要注意稀疏矩阵与实矩阵相乘的问题（区分`np.dot(x, W)`以及`x.dot(W)`）

2023.10.23补充：

1. 为了代码简洁，合并针对mnist和rcv1的模型

## sampling.py

封装数据类，数据类中存储了X, Y的值，训练集和测试集的划分，小批量采样等步骤

同时，在`sampling.py`中，我们定义了get_mnist以及get_rcv1函数，用于读取mnist和rcv1数据集

如果需要读取新的数据集，可以把这部分读取数据集的程序封装在这个地方

## federated_main.py

用于跑实验的代码，包括实验的参数设置（包括alpha， memory_length），选择哪一个算法，哪一个数据集

设置完参数后，调用中间的algorithm.alg_run完成联邦学习的实验

最后调用util中定义的excel_solver类，完成对实验得到的current_time, current_grad_times, current_loss, current_round的存储

## getParam.py

主要分为四个函数：`summary_csv`, `get_param`, `sum_up_param`, `generate_csv`

`get_params`: 为了能够并行化处理，在这里使用了multiprocessing，组合了dataset_name, algorithm_name, eta, times的会出现的情况，使用了starmap来并行化执行下面的generate_csv函数，同时starmap也传递提前生成的

times的设置是因为在相同的dataset_name, algorithm_name, eta的情况下，我们需要执行三次实验，来尽可能保证损失函数的准确性

`generate_csv`: 根据传递的dataset_name, algorithm_name, eta, times, 生成算法所需要的超参数，并执行相关算法，得到在预定最大评估次数下最后得到的损失函数

`summary_param`: 我们需要将上面得到的csv得到的损失函数值导入到程序中，因为我们每种情况都会执行三次，所以我们需要将每一种情况的三个csv导入到程序中，得到三个csv最后得到的损失函数，取中位数作为这一个eta的损失函数。然后，我们将相同dataset，相同algorithm的不同eta的损失函数进行比较。找出最优的eta

`sum_up_param`: 将各种dataset和各种algorithm得到的最优eta以及对应的损失函数导出一个csv



## notes:

### 后续任务

#### 第一次任务（ddl 10.30）

1. 加快调参代码运行速度，加入并行化程序
2. 程序选择最优eta参数
3. 完成文档编写
4. 控制每种情况的最大评估次数保持一致，方便后续进行对比
   
-> has finished.

----

#### 第二次任务

1. 对比算法（10.30后）
   
-> has finished.
   
   已添加FedAvg_SGD, FedAvg_GD, FedZO
   
2. 添加不同的数据集（11.10 start）
   
-> still working.

   已添加fashion_mnist
   
   正在添加cifar10
   
   需要注意梯度爆炸和梯度停滞，这时候可以注意客户端局部迭代数以及微分半径，微分半径r的设置与计算机浮点数的可识别范围k有关，如果是(f(x+r)-f(x-r))/(2r), r为k开三次方根; 如果是(f(x+r)-f(x))/(r), r为k开平方根

----

#### 第三次任务

   已完成cifar10添加

   正在添加svm模型

# Federated Learning Project

## algorithm.py

定义算法类，目前存储了FedAvg算法和Zeroth-grad算法

算法类的存储有固定格式，每个算法类都会固定分成四个函数

1. `__init__` 用于超参数和模型，数据集的存储，每一次的实验都务必要对模型进行重置

2. `update_weight` 用于模拟单个客户端的更新

3.  `average` 用于聚合

4. `alg_run`  负责整个算法的运行



## test.py

牛顿法验证代码，牛顿法精度较高，通过对比牛顿法的损失函数值和我们写的损失函数值即可判断代码的准确性

## 

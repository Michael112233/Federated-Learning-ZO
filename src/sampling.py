#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from sklearn.preprocessing import normalize
from scipy.sparse import hstack, csr_matrix

class mnist_dataset:
    def __init__(self, database):
        self.X = database['Z']
        self.Y = database['y']
        self.Y = (self.Y.astype(int) >= 5) * 1  # 将数字>=5样本设为正例，其他数字设为负例
        # 添加一列全为1的偏置项列
        self.X = np.hstack((self.X, np.ones((self.X.shape[0], 1))))
        # 归一化特征向量
        self.X = normalize(self.X, axis=1, norm='l2')
        # 划分训练集、测试集
        training_samples_ratio = 0.7
        N = len(self.Y)
        N_train = int(N * training_samples_ratio)

        rand_idxs = np.random.permutation(N)
        self.X_train = self.X[rand_idxs[:N_train]]
        self.Y_train = self.Y[rand_idxs[:N_train]]
        self.X_test = self.X[rand_idxs[N_train:]]
        self.Y_test = self.Y[rand_idxs[N_train:]]

    def sample(self, partition_index, batch_size=64):
        # print(self.X_train)
        # print(self.Y_train)
        # print(partition_index)
        # 随机选择一个小批量
        X_chosen = self.X_train[partition_index]
        Y_chosen = self.Y_train[partition_index]
        batch_indices = np.random.choice(len(X_chosen), batch_size, replace=False)
        X_batch = X_chosen[batch_indices]
        Y_batch = Y_chosen[batch_indices]
        return X_batch, Y_batch

    def full(self):
        return self.X_train, self.Y_train
    def length(self):
        return self.X_train.shape[0]

class csr_dataset:
    def __init__(self, x ,y):
        self.X = x
        self.Y = y
        # 创建全为1的列向量，作为偏置项列
        bias_column = np.ones(self.X.shape[0])

        # 将偏置项列转换为稀疏矩阵格式
        bias_column_sparse = csr_matrix(bias_column).transpose()

        # 将稀疏矩阵与偏置项列拼接
        self.X = hstack([self.X, bias_column_sparse])
        # 归一化特征向量
        self.X = normalize(self.X, axis=1, norm='l2')
        # 划分训练集、测试集
        training_samples_ratio = 0.7
        N = len(self.Y)
        N_train = int(N * training_samples_ratio)

        rand_idxs = np.random.permutation(N)
        self.X_train = self.X[rand_idxs[:N_train]]
        self.Y_train = self.Y[rand_idxs[:N_train]]
        self.X_test = self.X[rand_idxs[N_train:]]
        self.Y_test = self.Y[rand_idxs[N_train:]]

    def sample(self, partition_index, batch_size=64):
        # 随机选择一个小批量
        X_chosen = self.X_train[partition_index]
        Y_chosen = self.Y_train[partition_index]
        # print(X_chosen)
        batch_indices = np.random.choice(X_chosen.shape[0], batch_size, replace=False)
        X_batch = X_chosen[batch_indices]
        Y_batch = Y_chosen[batch_indices]
        return X_batch, Y_batch

    def full(self):
        return self.X_train, self.Y_train

    def length(self):
        return self.X_train.shape[0]

def iid_partition(dataset, num_clients):
    num_items = int(dataset.length() / num_clients)
    dict_clients, all_idxs = {}, [i for i in range(dataset.length())]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
        dict_clients[i] = list(dict_clients[i])
    return dict_clients

# def mnist_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items,
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users
#
#
# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     # 60,000 training imgs -->  200 imgs/shard X 300 shards
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # divide and assign 2 shards/client
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users
#
#
# def mnist_noniid_unequal(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset s.t clients
#     have unequal amount of data
#     :param dataset:
#     :param num_users:
#     :returns a dict of clients with each clients assigned certain
#     number of training imgs
#     """
#     # 60,000 training imgs --> 50 imgs/shard X 1200 shards
#     num_shards, num_imgs = 1200, 50
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # Minimum and maximum shards assigned per client:
#     min_shard = 1
#     max_shard = 30
#
#     # Divide the shards into random chunks for every client
#     # s.t the sum of these chunks = num_shards
#     random_shard_size = np.random.randint(min_shard, max_shard+1,
#                                           size=num_users)
#     random_shard_size = np.around(random_shard_size /
#                                   sum(random_shard_size) * num_shards)
#     random_shard_size = random_shard_size.astype(int)
#
#     # Assign the shards randomly to each client
#     if sum(random_shard_size) > num_shards:
#
#         for i in range(num_users):
#             # First assign each client 1 shard to ensure every client has
#             # atleast one shard of data
#             rand_set = set(np.random.choice(idx_shard, 1, replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)
#
#         random_shard_size = random_shard_size-1
#
#         # Next, randomly assign the remaining shards
#         for i in range(num_users):
#             if len(idx_shard) == 0:
#                 continue
#             shard_size = random_shard_size[i]
#             if shard_size > len(idx_shard):
#                 shard_size = len(idx_shard)
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)
#     else:
#
#         for i in range(num_users):
#             shard_size = random_shard_size[i]
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)
#
#         if len(idx_shard) > 0:
#             # Add the leftover shards to the client with minimum images:
#             shard_size = len(idx_shard)
#             # Add the remaining shard to the client with lowest data
#             k = min(dict_users, key=lambda x: len(dict_users.get(x)))
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[k] = np.concatenate(
#                     (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)
#
#     return dict_users
#
#
# def cifar_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items,
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users
#
#
# def cifar_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 250
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     # labels = dataset.train_labels.numpy()
#     labels = np.array(dataset.train_labels)
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users
#
#
# if __name__ == '__main__':
#     dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.1307,),
#                                                             (0.3081,))
#                                    ]))
#     num = 100
#     d = mnist_noniid(dataset_train, num)

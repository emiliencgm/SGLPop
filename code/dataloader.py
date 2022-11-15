"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import random
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
import numpy as np
#import pandas as pd
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
#自定义的数据集继承自Dataset，必须重写的方法:__init__()读取数据, __len__()自定义数据集的长度, __getitem__()对自定义数据集进行索引
#用Dataset制作好数据集后交给DataLoader可以自动输出每个batch的数据及标签
import world
from world import cprint
from time import time


class Loader(Dataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla\ yelp2018\ amazon-book\ MIND dataset
    """

    def __init__(self, config = world.config, path="../data/yelp2018"):
        # train or test
        cprint(f'loading [{path}]')
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        valid_file = path + '/valid.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validDataSize = 0

        #================Pop=================#
        TrainTestPop = {}
        TrainTestPop_user = {}
        #================Pop=================#

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
                    #================Pop=================#
                    for item in items:
                        if item in TrainTestPop.keys():
                            TrainTestPop[item] += 1
                        else:
                            TrainTestPop[item] = 1
                    if uid in TrainTestPop_user.keys():
                        TrainTestPop_user[uid] += len(items)
                    else:
                        TrainTestPop_user[uid] = len(items)
                    #================Pop=================#
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if l[1]:#TODO 什么目的？？第一个item为0怎么办？
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        testUniqueUsers.append(uid)
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.n_user = max(self.n_user, uid)
                        self.testDataSize += len(items)
                        #================Pop=================#
                        for item in items:
                            if item in TrainTestPop.keys():
                                TrainTestPop[item] += 1
                            else:
                                TrainTestPop[item] = 1
                        if uid in TrainTestPop_user.keys():
                            TrainTestPop_user[uid] += len(items)
                        else:
                            TrainTestPop_user[uid] = len(items)
                        #================Pop=================#
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        
        if os.path.exists(valid_file):#TODO 事实上并没有用验证集？！
            with open(valid_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        if l[1]:
                            items = [int(i) for i in l[1:]]
                            uid = int(l[0])
                            validUniqueUsers.append(uid)
                            validUser.extend([uid] * len(items))
                            validItem.extend(items)
                            self.m_item = max(self.m_item, max(items))
                            self.n_user = max(self.n_user, uid)
                            self.validDataSize += len(items)
                            #================Pop=================#
                            for item in items:
                                if item in TrainTestPop.keys():
                                    TrainTestPop[item] += 1
                                else:
                                    TrainTestPop[item] = 1
                            if uid in TrainTestPop_user.keys():
                                TrainTestPop_user[uid] += len(items)
                            else:
                                TrainTestPop_user[uid] = len(items)
                            #================Pop=================#
            self.validUniqueUsers = np.array(validUniqueUsers)
            self.validUser = np.array(validUser)
            self.validItem = np.array(validItem)
        
        self.m_item += 1
        self.n_user += 1

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{config['dataset']} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")#针对无验证集时的稀疏度计算公式

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_user, self.m_item))
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()

        #================Pop=================#
        self._itemPop_dict, self._reverse_itemPop_dict, self._testDict_pop = self.__build_pop(TrainTestPop)
        self._userPop_dict, self._reverse_userPop_dict = self.__build_pop_user(TrainTestPop_user)
        self._TrainTestPop = TrainTestPop
        self._item_pop_label = []
        self._user_pop_label = []
        for item in range(self.m_item):
            self._item_pop_label.append(self._reverse_itemPop_dict[item])
        self._item_pop_label = np.array(self._item_pop_label)
        for user in range(self.n_user):
            self._user_pop_label.append(self._reverse_userPop_dict[user])
        self._user_pop_label = np.array(self._user_pop_label)
        #================Pop=================#

        print(f"{config['dataset']} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    #================Pop=================#
    @property
    def itemPop_dict(self):
        '''
        {0 : tensor[items in pop0] }
        '''
        return self._itemPop_dict

    @property
    def reverse_itemPop_dict(self):
        '''
        {item : group}
        '''
        return self._reverse_itemPop_dict

    @property
    def item_pop_label(self):
        '''
        np.array([pop1, pop2, ...])
        '''
        return self._item_pop_label

    @property
    def user_pop_label(self):
        '''
        np.array([pop1, pop2, ...])
        '''
        return self._user_pop_label

    @property
    def testDict_pop(self):
        '''
        {\n
         0 : {user:[items in pop0]}\n
         1 : {user:[items in pop1]}\n
        }
        '''
        return self._testDict_pop

    @property
    def TrainTestPop(self):
        return self._TrainTestPop
    #================Pop=================#
    #================Pop=================#
    def __build_pop(self, TrainTestPop):
        num_group = world.config['pop_group']
        item_per_group = int(self.m_items / num_group)
        TrainTestPopSorted = sorted(TrainTestPop.items(), key=lambda x: x[1])
        ItemPopGroupDict = {}#查询分组中有哪些item的字典
        testDict_PopGroup = {}#查询不同分组下用户在Test集中交互过的item的字典
        reverse_ItemPopGroupDict = {}#查询item属于哪个分组的字典
        #按照Pop分组，并存储至字典[0=Cold, 9=Hot]
        for group in range(num_group):
            ItemPopGroupDict[group] = []
            if group == num_group-1:
                for item, _ in TrainTestPopSorted[group * item_per_group:]:
                    ItemPopGroupDict[group].append(item)
            else:
                for item, _ in TrainTestPopSorted[group * item_per_group: (group+1) * item_per_group]:
                    ItemPopGroupDict[group].append(item)
        #转换为tensor格式
        for group, items in ItemPopGroupDict.items():
            ItemPopGroupDict[group] = torch.tensor(items)
        
        #生成查询item属于哪个分组的字典
        for item in range(self.m_items):
            group = -1
            for i in range(num_group):
                if item in ItemPopGroupDict[i]:
                    group = i
                    break
            reverse_ItemPopGroupDict[item] = group

        #初始化testDict_PopGroup的格式：testDict_PopGroup={0:{user:ColdItem}}
        for group in range(num_group):
            testDict_PopGroup[group] = {}
        #生成不同热度分组下的用户test交互item字典
        for user, items in self.testDict.items():                
            Hot = {}
            for group in range(num_group):
                Hot[group] = []
            for item in items:
                group = reverse_ItemPopGroupDict[item]
                Hot[group].append(item)
            for group in range(num_group):
                if Hot[group]:
                    testDict_PopGroup[group][user] = Hot[group]
                else:
                    testDict_PopGroup[group][user] = [999999999999999]#缺省值
        #print(testDict_PopGroup[0])
        return ItemPopGroupDict, reverse_ItemPopGroupDict, testDict_PopGroup
    
    def __build_pop_user(self, TrainTestPop_user):
        '''
        TrainTestPop_user
        '''
        num_group = world.config['pop_group']
        user_per_group = int(self.n_users / num_group)
        TrainTestPopSorted = sorted(TrainTestPop_user.items(), key=lambda x: x[1])
        UserPopGroupDict = {}#查询分组中有哪些item的字典
        reverse_UserPopGroupDict = {}#查询item属于哪个分组的字典
        #按照Pop分组，并存储至字典[0=Cold, 9=Hot]
        for group in range(num_group):
            UserPopGroupDict[group] = []
            if group == num_group-1:
                for user, _ in TrainTestPopSorted[group * user_per_group:]:
                    UserPopGroupDict[group].append(user)
            else:
                for user, _ in TrainTestPopSorted[group * user_per_group: (group+1) * user_per_group]:
                    UserPopGroupDict[group].append(user)
        #转换为tensor格式
        for group, users in UserPopGroupDict.items():
            UserPopGroupDict[group] = torch.tensor(users)
        
        #生成查询item属于哪个分组的字典
        for user in range(self.n_users):
            group = -1
            for i in range(num_group):
                if user in UserPopGroupDict[i]:
                    group = i
                    break
            reverse_UserPopGroupDict[user] = group

        
        return UserPopGroupDict, reverse_UserPopGroupDict

    #================Pop=================#


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        """
        Graph = \n
        D^(-1/2) @ A @ D^(-1/2) \n
        A = \n
        |0,   R|\n
        |R.T, 0|\n
        度归一化相当于对user-item的交互R矩阵中u和i的位置乘上 1/sqrt(|u|) * 1/sqrt(|i|) |x|指节点x的邻居数
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix --- All train data")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                #此处会显存爆炸
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0]) TODO 无自连接
                
                rowsum = np.array(adj_mat.sum(axis=1))#度
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)#对角阵
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            if world.config['if_big_matrix']:
                print(f"split the adj matrix to {world.config['n_fold']} folds")
                self.Graph = self._split_matrix(norm_adj)
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def _split_matrix(self, norm_adj):
        norm_adj_split = []
        fold_len = (self.n_user + self.m_items) // world.config['n_fold']
        for i_fold in range(world.config['n_fold']):
            start = i_fold * fold_len
            if i_fold == world.config['n_fold']-1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            norm_adj_split.append(self._convert_sp_mat_to_sp_tensor(norm_adj[start:end]).coalesce().to(world.device))
        return norm_adj_split

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems

    # train loader and sampler part
    '''
    Dataset的所有子类都应该重写方法__len__(), __getitem__()
    '''
    def __len__(self):
        return self.traindataSize

    def __getitem__(self, idx):
        '''
        input: user在trainUser列表中的idx
        output: 随机三元组(user, pos, neg)
        '''
        user = self.trainUser[idx]
        pos = random.choice(self._allPos[user])
        while True:
            neg = np.random.randint(0, self.m_item)
            if neg in self._allPos[user]:
                continue
            else:
                break
        return user, pos, neg
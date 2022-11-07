#from torch_sparse.tensor import to
import scipy.sparse as sp
import torch
import torch.nn as nn
from utils import randint_choice
import numpy as np
import torch.nn.functional as F
import world
import math

class Contrast(nn.Module):
    def __init__(self, gcn_model, tau=world.config['temp_tau']):
        super(Contrast, self).__init__()
        self.gcn_model = gcn_model
        self.tau = tau
        self.itemAugProbDict, self.dropItems, self.addItems, self.trainItemProb = None, None, None, None

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        '''
        计算一个z1和一个z2两个向量的相似度/或者一个z1和多个z2的各自相似度。
        即两个输入的向量数（行数）可能不同。
        '''
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1,z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())
            #TODO 这不是相似度！仅仅是余弦值，可以小于零！ cos(z1,z2) = z1*z2/|z1||z2|

    def info_nce_loss_overall(self, z1, z2, z_all):
        '''
        即式(8)\n
        z1--z2: pos,  z_all: neg\n
        return: InfoNCEloss      
        '''
        f = lambda x: torch.exp(x / self.tau)
        # batch_size
        between_sim = f(self.sim(z1, z2))
        # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        all_sim = f(self.sim(z1, z_all))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
        return loss


    def Edge_drop_random(self, p_drop, p_e_drop, p_e_add):
        '''
        SGL1:对于采样要删除的边，冷门物品的边不删除，反而增加一条边\n
        SGL2:对于采样要删除的边，热门物品的边不删除; 冷门物品的边不删除，反而增加一条边\n
        return: dropout后保留的交互构成的按度归一化的邻接矩阵(sparse)
        '''
        n_nodes = self.gcn_model.num_users + self.gcn_model.num_items
        #================Pop=================#
        #注意数组复制问题！
        trainUser = self.gcn_model.dataset.trainUser.copy()
        trainItem = self.gcn_model.dataset.trainItem.copy()
        if world.config['if_pop']:
            if world.config['pop_mode'] in ['SGL3', 'SGL4', 'SGL5']:
                itemAugProbDict, dropItems, addItems, trainItemProb = self.itemAugProbDict, self.dropItems, self.addItems, self.trainItemProb
                user_np, item_np = self.getItemAug(trainItemProb, itemAugProbDict, addItems, p_e_drop, p_e_add)
            else:
                raise TypeError('pop_mode')
        else:
            keep_idx = randint_choice(len(self.gcn_model.dataset.trainUser), size=int(len(self.gcn_model.dataset.trainUser) * (1 - p_drop)), replace=False)
            user_np = trainUser[keep_idx]
            item_np = trainItem[keep_idx]
        #================Pop=================#
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.gcn_model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        if world.config['if_big_matrix']:
            g = self.gcn_model.dataset._split_matrix(adj_matrix)
            for fold in g:
                fold.requires_grad = False
        else:
            #coo = adj_matrix.tocoo().astype(np.float32)
            #row = torch.Tensor(coo.row).long()
            #col = torch.Tensor(coo.col).long()
            #index = torch.stack([row, col])
            #data = torch.FloatTensor(coo.data)
            #g = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce().to(world.device)
            g = self.gcn_model.dataset._convert_sp_mat_to_sp_tensor(adj_matrix).coalesce().to(world.device)
            g.requires_grad = False
        return g
    
    def getItemAugProb(self):
        #生成每个物品对应的数据增强概率——对热度自适应
        print('pre-calculate item augmentation probability')
        ItemAugProbDict = {}
        ItemPopDict = self.gcn_model.TrainTestPop
        meanLog = 0
        maxLog = 0
        minLog = 1000
        dropItems = []
        addItems = []

        for item, pop in ItemPopDict.items():
            log = math.log(pop)
            ItemAugProbDict[item] = log
            meanLog += log
            maxLog = max(maxLog, log)
            minLog = min(minLog, log)
        meanLog = meanLog / len(ItemPopDict)

        if world.config['pop_mode'] == 'SGL3':
            for item, log in ItemAugProbDict.items():
                prob = (log - meanLog) / (maxLog - meanLog)
                ItemAugProbDict[item] = prob
                if prob < 0:
                    addItems.append(item)
                else:
                    dropItems.append(item)

        elif world.config['pop_mode'] == 'SGL4':
            for item, log in ItemAugProbDict.items():
                if log >= meanLog:# prob = dropout prob
                    dropItems.append(item)
                    prob = (maxLog - log) / (maxLog - meanLog)
                    ItemAugProbDict[item] = prob
                else:#final prob < 0 --> - add prob
                    addItems.append(item)
                    prob = - (meanLog - log) / (meanLog - minLog)
                    ItemAugProbDict[item] = prob

        elif world.config['pop_mode'] == 'SGL5':
            for item, log in ItemAugProbDict.items():
                if log >= meanLog:# prob = dropout prob
                    dropItems.append(item)
                    prob = (log - meanLog) / (maxLog - meanLog)
                    ItemAugProbDict[item] = prob
                else:#final prob < 0 --> - add prob
                    addItems.append(item)
                    prob = - (meanLog - log) / (meanLog - minLog)
                    ItemAugProbDict[item] = prob
        else:
            raise TypeError('Pop_Mode')
        print('Hot items:',len(dropItems),'Cold items:',len(addItems), 'Total items:',len(ItemAugProbDict))
        trainItemProb = self.gcn_model.dataset.trainItem.copy()
        
        for idx in range(len(trainItemProb)):
            item = trainItemProb[idx]
            trainItemProb[idx] = ItemAugProbDict[item]        
            
        self.itemAugProbDict, self.dropItems, self.addItems, self.trainItemProb = ItemAugProbDict, dropItems, addItems, trainItemProb

        print('finished')
        return ItemAugProbDict, dropItems, addItems, trainItemProb
    
    def getItemAug(self, trainItemProb, ItemAugProbDict, addItems, p_e_drop, p_e_add):
        #根据trainItemProb,dropout一些边，并add一些边
        #热门的dropout概率高？冷门（<0）的一定被留下，后续再为冷门增加边
        print('generating one view')
        if world.config['pop_mode'] == 'SGL3':
            trainItemDropProb = torch.tensor(trainItemProb, dtype=torch.float) * p_e_drop
            trainItemDropProb = torch.where(trainItemDropProb >= 0., trainItemDropProb, torch.zeros_like(trainItemDropProb))
            trainItemKeep = torch.bernoulli(1. - trainItemDropProb).to(torch.bool)#TODO 对Hot的交互dropout

            trainUser = self.gcn_model.dataset.trainUser.copy()
            trainItem = self.gcn_model.dataset.trainItem.copy()
            trainUser, trainItem = trainUser[trainItemKeep], trainItem[trainItemKeep]
            add_Users = []
            add_Items = []
        
            for item in addItems:
                for edge in range(self.gcn_model.TrainTestPop[item]):
                    if torch.bernoulli(torch.tensor(-ItemAugProbDict[item] * p_e_add)).to(torch.bool):
                        addUser = randint_choice(self.gcn_model.num_users, size=1,replace=False)[0]
                        add_Users.append(addUser)
                        add_Items.append(item)
        
        elif world.config['pop_mode'] in ['SGL4', 'SGL5']:

            trainItemDropProb = torch.tensor(trainItemProb, dtype=torch.float) * p_e_drop
            trainItemDropProb = torch.where(trainItemDropProb >= 0., trainItemDropProb, torch.zeros_like(trainItemDropProb))
            trainItemKeep = torch.bernoulli(1. - trainItemDropProb).to(torch.bool)

            trainUser = self.gcn_model.dataset.trainUser[trainItemKeep]
            trainItem = self.gcn_model.dataset.trainItem[trainItemKeep]
            
            #TODO 此处执行效率过低
            add_Users = []
            add_Items = []
            #for item in addItems:
            #    for i_edge in range(self.gcn_model.TrainTestPop[item]):
            #        if torch.bernoulli(torch.tensor(-ItemAugProbDict[item] * p_e_add)).to(torch.bool):
            #            addUser = randint_choice(self.gcn_model.num_users, size=1,replace=False)[0]
            #            add_Users.append(addUser)
            #            add_Items.append(item)
            Addidx = []
            for item in addItems:
                Addidx.extend([-ItemAugProbDict[item] * p_e_add]*self.gcn_model.TrainTestPop[item])
                add_Items.extend([item]*self.gcn_model.TrainTestPop[item])
            Addidx = torch.bernoulli(torch.tensor(Addidx)).to(torch.bool)
            add_Users = randint_choice(self.gcn_model.num_users, size=len(Addidx), replace=True)[Addidx]#随机add的用户可以重复
            add_Items = np.array(add_Items)[Addidx]
        else:
            raise TypeError('Pop_Mode')
        print('one view is generated')
        return np.append(trainUser, add_Users), np.append(trainItem, add_Items)


    def Random_walk_random(self, p_drop, p_e_drop, p_e_add):
        '''
        直接调用num_layers次Edge_drop_random生成每一层的图矩阵
        '''
        UIViews = []
        for layer in range(world.config['num_layers']):
            UIViews.append(self.Edge_drop_random(p_drop, p_e_drop, p_e_add))
        return UIViews

    def Node_drop_random(self, node_drop_prob):
        n_nodes = self.gcn_model.num_users + self.gcn_model.num_items
        

        return 0
    
    def Feature_mask_random(self, feature_mask_prob):
        num_features = world.config['latent_dim_rec']#对于推荐系统，无输入属性，仅仅是embedding lookup
        return 0
    
    def get_views(self):
        '''
        return: 2 UI-Views
        '''
        if world.config['model'] == 'SGL-ED':#边dropout
            if (world.config['pop_mode'] in ['SGL3', 'SGL4', 'SGL5']) and world.config['if_pop']:
                if self.itemAugProbDict is None or self.dropItems is None or self.addItems is None or self.trainItemProb is None:
                    self.getItemAugProb()          
            UIv1 = self.Edge_drop_random(world.config['edge_drop_prob'], world.config['P_e_drop1'], world.config['P_e_add1'])
            UIv2 = self.Edge_drop_random(world.config['edge_drop_prob'], world.config['P_e_drop2'], world.config['P_e_add2'])
        elif world.config['model'] == 'SGL-ND':#节点droopout          
            UIv1 = self.Node_drop_random(world.config['edge_drop_prob'])
            UIv2 = self.Node_drop_random(world.config['edge_drop_prob'])
        elif world.config['model'] == 'SGL-RW':#随机游走——每层GCN对应一个对比视图
            if (world.config['pop_mode'] in ['SGL3', 'SGL4', 'SGL5']) and world.config['if_pop']:
                if self.itemAugProbDict is None or self.dropItems is None or self.addItems is None or self.trainItemProb is None:
                    self.getItemAugProb()
            UIv1 = self.Random_walk_random(world.config['edge_drop_prob'], world.config['P_e_drop1'], world.config['P_e_add1'])
            UIv2 = self.Random_walk_random(world.config['edge_drop_prob'], world.config['P_e_drop2'], world.config['P_e_add2'])
        else:
            raise TypeError('Pop_Mode')
        Views = {'uiv1': UIv1, 'uiv2':UIv2}
        return Views

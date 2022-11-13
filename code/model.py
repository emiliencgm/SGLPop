"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import os
import world
import torch
from dataloader import Loader
from torch import nn
import numpy as np
import torch.nn.functional as F

    

class SGL_ED(nn.Module):
    def __init__(self, config:dict, dataset:Loader):
        super(SGL_ED, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("user:{}, item:{}".format(self.num_users, self.num_items))
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['num_layers']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if self.config['if_pretrain'] == 0:
            world.cprint('use NORMAL distribution UI for Embedding')
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined Embedding')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

        print(f"SGL is ready to go!")

    #================Pop=================#
    @property
    def itemPop_dict(self):
        return self.dataset.itemPop_dict

    @property
    def testDict_pop(self):
        return self.dataset.testDict_pop

    @property
    def reverse_itemPop_dict(self):
        return self.dataset.reverse_itemPop_dict


    @property
    def TrainTestPop(self):
        return self.dataset.TrainTestPop
    #================Pop=================#
    #def __dropout_x(self, x, keep_prob):
    #    size = x.size()
    #    index = x.indices().t()
    #    values = x.values()
    #    random_index = torch.rand(len(values)) + keep_prob
    #    random_index = random_index.int().bool()
    #    index = index[random_index]
    #    values = values[random_index]/keep_prob
    #    g = torch.sparse.FloatTensor(index.t(), values, size)
    #    return g
    
    #def __dropout(self, keep_prob):
    #    if self.A_split:
    #        graph = []
    #        for g in self.Graph:
    #           graph.append(self.__dropout_x(g, keep_prob))
    #    else:
    #        graph = self.__dropout_x(self.Graph, keep_prob)
    #    return graph


    def view_computer(self, g_droped):
        """
        在g_droped上用LightGCN传播user和item的信息。
        return: 用于推荐的user和item的嵌入。
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if world.config['model']=='SGL-RW':
            for layer in range(self.n_layers):
                if world.config['if_big_matrix']:
                    temp_emb = []
                    for i_fold in range(len(g_droped[layer])):
                        temp_emb.append(torch.sparse.mm(g_droped[layer][i_fold], all_emb))
                    all_emb = torch.cat(temp_emb, dim=0)
                else:
                    all_emb = torch.sparse.mm(g_droped[layer], all_emb)
                embs.append(all_emb)
        elif world.config['model']=='SGL-ED':
            for layer in range(self.n_layers):
                if world.config['if_big_matrix']:
                    temp_emb = []
                    for i_fold in range(len(g_droped)):
                        temp_emb.append(torch.sparse.mm(g_droped[i_fold], all_emb))
                    all_emb = torch.cat(temp_emb, dim=0)
                else:
                    all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


    def computer(self):
        """
        原始LightGCN,应该不使用Dropout
        return: 用于推荐的user和item的嵌入。
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        #if self.config['dropout']:
        #    if self.training:
        #        g_droped = self.__dropout(self.keep_prob)
        #    else:
        #        g_droped = self.Graph        
        #else:
        #   g_droped = self.Graph
        g_droped = self.Graph    
        for layer in range(self.n_layers):
            if world.config['if_big_matrix']:
                temp_emb = []
                for i_fold in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[i_fold], all_emb))
                all_emb = torch.cat(temp_emb, dim=0)
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    
    def getUsersRating(self, users):
        '''
        先执行一次model.computer().
        return rating=指定users对每个item做内积后过Sigmoid()
        '''
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))#TODO 内积后过Sigmoid()作为输出Rating
        return rating

    #================Pop=================#
    def getItemRating(self):
        '''
        获取输入items1, items2对全部user的平均得分
        return: rating1=Hot, rating2=Cold
        '''
        itemsPopDict = self.itemPop_dict
        all_users, all_items = self.computer()
        items_embDict = {}
        for group in range(world.config['pop_group']):
            items_embDict[group] = all_items[itemsPopDict[group].long()]
        users_emb = all_users
        #rating = self.f(torch.matmul(items_emb, users_emb.t()))#TODO 内积后过Sigmoid()作为输出Rating
        rating_Dict = {}
        for group in range(world.config['pop_group']):
            rating_Dict[group] = torch.matmul(items_embDict[group], users_emb.t())
            rating_Dict[group] = torch.mean(rating_Dict[group], dim=1)
            rating_Dict[group] = torch.mean(rating_Dict[group])
        return rating_Dict
    #================Pop=================#

    def getEmbedding(self, users, pos_items, neg_items):
        '''
        先执行一次model.computer().
        return: users, pos_items, neg_items各自的初始embedding(item在聚合KG信息前的embedding)和LightGCN更新后的embedding
        '''
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        '''
        输入一个batch的users、pos_items、neg_items
        reG_loss = users、pos_items、neg_items初始embedding的L2正则化loss
        reC_loss = Σ{ softplus[ (ui,negi) - (ui,posi) ] }
        '''
        (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        '''
        pos_scores=
        [(u1,pos1), (u2,pos2), ...]
        '''
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        '''
        neg_scores=
        [(u1,neg1), (u2,neg2), ...]
        '''
        # mean or sum
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))#TODO SOFTPLUS()!!!
        if(torch.isnan(loss).any().tolist()):
            print("user emb")
            print(userEmb0)
            print("pos_emb")
            print(posEmb0)
            print("neg_emb")
            print(negEmb0)
            print("neg_scores")
            print(neg_scores)
            print("pos_scores")
            print(pos_scores)
            return None
        return loss, reg_loss
       

    #def forward(self, users, items):
    #    # compute embedding
    #    all_users, all_items = self.computer()
    #    # print('forward')
    #    #all_users, all_items = self.computer()
    #    users_emb = all_users[users]
    #    items_emb = all_items[items]
    #    inner_pro = torch.mul(users_emb, items_emb)
    #    gamma     = torch.sum(inner_pro, dim=1)
    #    return gamma



class SGL_RW(nn.Module):
    def __init__(self, config:dict, dataset:Loader):
        super(SGL_RW, self).__init__()
        self.config = config
        self.dataset = dataset
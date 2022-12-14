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
            if world.config['init_method'] == 'Normal':
                world.cprint('use NORMAL distribution UI for Embedding')
                nn.init.normal_(self.embedding_user.weight, std=0.1)
                nn.init.normal_(self.embedding_item.weight, std=0.1)
            elif world.config['init_method'] == 'Xavier':
                world.cprint('use Xavier_uniform distribution UI for Embedding')
                nn.init.xavier_uniform_(self.embedding_user.weight, gain=1.0)
                nn.init.xavier_uniform_(self.embedding_item.weight, gain=1.0)
            else:
                raise TypeError('init method')
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
        ???g_droped??????LightGCN??????user???item????????????
        return: ???????????????user???item????????????
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        if world.config['model'] in ['SGL-ED', 'SGL-RW']:
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
        elif world.config['model'] in ['SimGCL']:
            embs = []#SimGCL???????????????????????????
            if world.config['model']=='SimGCL':
                for layer in range(self.n_layers):
                    if world.config['if_big_matrix']:
                        temp_emb = []
                        for i_fold in range(len(g_droped)):
                            temp_emb.append(torch.sparse.mm(g_droped[i_fold], all_emb))
                        all_emb = torch.cat(temp_emb, dim=0)
                    else:
                        all_emb = torch.sparse.mm(g_droped, all_emb)
                    '''
                    with torch.no_grad():
                        low = torch.zeros_like(all_emb).float()
                        high = torch.ones_like(all_emb).float()
                        random_noise = torch.distributions.uniform.Uniform(low, high).sample()
                        noise = torch.mul(torch.sign(all_emb),torch.nn.functional.normalize(random_noise, dim=1)) * world.config['eps_SimGCL']
                    '''
                    low = torch.zeros_like(all_emb).float()
                    high = torch.ones_like(all_emb).float()
                    random_noise = torch.distributions.uniform.Uniform(low, high).sample()
                    noise = torch.mul(torch.sign(all_emb),torch.nn.functional.normalize(random_noise, dim=1)) * world.config['eps_SimGCL']
                    all_emb += noise
                    embs.append(all_emb)
            else:
                raise TypeError('model-mode')
        else:
            raise TypeError('model-mode')
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


    def computer(self):
        """
        ??????LightGCN,???????????????Dropout
        return: ???????????????user???item????????????
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
        ???????????????model.computer().
        return rating=??????users?????????item???????????????Sigmoid()
        '''
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))#TODO ????????????Sigmoid()????????????Rating
        return rating

    #================Pop=================#
    def getItemRating(self):
        '''
        ????????????items1, items2?????????user???????????????
        return: rating1=Hot, rating2=Cold
        '''
        itemsPopDict = self.itemPop_dict
        all_users, all_items = self.computer()
        items_embDict = {}
        for group in range(world.config['pop_group']):
            items_embDict[group] = all_items[itemsPopDict[group].long()]
        users_emb = all_users
        #rating = self.f(torch.matmul(items_emb, users_emb.t()))#TODO ????????????Sigmoid()????????????Rating
        rating_Dict = {}
        for group in range(world.config['pop_group']):
            rating_Dict[group] = torch.matmul(items_embDict[group], users_emb.t())
            rating_Dict[group] = torch.mean(rating_Dict[group], dim=1)
            rating_Dict[group] = torch.mean(rating_Dict[group])
        return rating_Dict
    #================Pop=================#

    def getEmbedding(self, users, pos_items, neg_items):
        '''
        ???????????????model.computer().
        return: users, pos_items, neg_items???????????????embedding(item?????????KG????????????embedding)???LightGCN????????????embedding
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
        ????????????batch???users???pos_items???neg_items
        reG_loss = users???pos_items???neg_items??????embedding???L2?????????loss
        reC_loss = ??{ softplus[ (ui,negi) - (ui,posi) ] }
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
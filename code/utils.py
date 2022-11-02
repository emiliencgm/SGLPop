'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import collections
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import Loader
from time import time
from sklearn.metrics import roc_auc_score
import random
import os
sample_ext = False
# try:
#     from cppimport import imp_from_filepath
#     from os.path import join, dirname
#     path = join(dirname(__file__), "sources/sampling.cpp")
#     sampling = imp_from_filepath(path)
#     sampling.seed(world.seed)
#     sample_ext = True
# except:
#     world.cprint("Cpp extension not loaded")
#     sample_ext = False

def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).
    """
    a = np.arange(high)
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = p.flatten()
        p[exclusion] = 0
    if p is not None:
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample


def _L2_loss_mean(x):
    '''
    对x每个行向量求2范数的平方,各行求平均/2
    return:scalar
    '''
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class BPRLoss:
    def __init__(self, recmodel, opt):
        self.model = recmodel
        self.opt = opt
        self.weight_decay = world.config["weight_decay"]

    def compute(self, users, pos, neg):
        '''
        return: bpr_loss + weight_devcay*reg_loss\n
        并不更新参数
        '''
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss
        return loss

    def stageOne(self, users, pos, neg):
        '''
        return: bpr_loss + weight_decay*reg_loss\n
        并更新参数
        '''
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    file = f"{world.config['model']}-{world.config['dataset']}-dim{world.config['latent_dim_rec']}-dropProb{world.config['edge_drop_prob']}-tau{world.config['temp_tau']}.pth.tar"
    return os.path.join(world.FILE_PATH,file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(groundTrues, groundTrues_popDict, r, r_popDict, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    right_pred_popDict = {}
    num_group = world.config['pop_group']
    for group in range(num_group):
        right_pred_popDict[group] = r_popDict[group][:, :k].sum(1)

    precis_n = k
    recall_n = np.array([len(groundTrues[i]) for i in range(len(groundTrues))])

    recall_n_popDict = {}
    for group in range(num_group):
        recall_n_popDict[group] = np.array([len(groundTrues_popDict[group][i]) for i in range(len(groundTrues_popDict[group]))])

    recall = np.sum(right_pred/recall_n)
    recall_popDict = {}
    recall_Contribute_popDict = {}
    for group in range(num_group):
        recall_popDict[group] = np.sum(right_pred_popDict[group]/recall_n_popDict[group])
        recall_Contribute_popDict[group] = np.sum(right_pred_popDict[group]/recall_n)

    precis = np.sum(right_pred)/precis_n

    return {'recall': recall, 'recall_popDIct': recall_popDict, 'recall_Contribute_popDict': recall_Contribute_popDict, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(groundTrues, groundTrue_popDicts, pred_data):
    r = []
    #================Pop=================#
    r_pop = {}
    for group in range(world.config['pop_group']):
        r_pop[group] = []

    for user in range(len(groundTrues)):
        groundTrue = groundTrues[user]
        groundTrue_popDict = {}
        for group in range(world.config['pop_group']):
            groundTrue_popDict[group] = groundTrue_popDicts[group][user]
        predictTopK = pred_data[user]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
        pred_pop = {}
        for group in range(world.config['pop_group']):
            pred_pop[group] = list(map(lambda x: x in groundTrue_popDict[group], predictTopK))
            pred_pop[group] = np.array(pred_pop[group]).astype("float")
            r_pop[group].append(pred_pop[group])
    for group in range(world.config['pop_group']):
        r_pop[group] = np.array(r_pop[group]).astype("float")
    #================Pop=================#
    return np.array(r).astype('float'), r_pop

# ====================end Metrics=============================
# =========================================================

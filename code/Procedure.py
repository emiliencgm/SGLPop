'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
from concurrent.futures.thread import _worker
from contrast import Contrast
from torch.utils.data.dataloader import DataLoader
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score

CORES = multiprocessing.cpu_count() // 2


def BPR_train_contrast(dataset, recommend_model, loss_class, contrast_model :Contrast, contrast_views, epoch, optimizer, neg_k=1, w=None):
    Recmodel :model.SGL_ED = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    batch_size = world.config['bpr_batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)#每个batch为batch_size对(user, pos_item, neg_item), 见Dataset.__getitem__

    total_batch = len(dataloader)
    aver_loss = 0.
    aver_loss_main = 0.
    aver_loss_ssl = 0.
    # For SGL
    uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
    for batch_i, train_data in tqdm(enumerate(dataloader), total=len(dataloader),disable=True):
        batch_users = train_data[0].long().to(world.device)
        batch_pos = train_data[1].long().to(world.device)
        batch_neg = train_data[2].long().to(world.device)

        # main task (batch based)
        # bpr loss for a batch of users
        l_main = bpr.compute(batch_users, batch_pos, batch_neg)#bpr_loss + λ2*reg_loss
        l_ssl = list()#λ1*InfoNCEloss
        items = batch_pos # [B*1]

        usersv1, itemsv1 = Recmodel.view_computer(uiv1)
        usersv2, itemsv2 = Recmodel.view_computer(uiv2)

        # from SGL source
        items_uiv1 = itemsv1[items]
        items_uiv2 = itemsv2[items]
        l_item = contrast_model.info_nce_loss_overall(items_uiv1, items_uiv2, itemsv2)

        users = batch_users
        users_uiv1 = usersv1[users]
        users_uiv2 = usersv2[users]
        l_user = contrast_model.info_nce_loss_overall(users_uiv1, users_uiv2, usersv2)
        l_ssl.extend([l_user*world.config['lambda1'], l_item*world.config['lambda1']])
        
        if l_ssl:
            l_ssl = torch.stack(l_ssl).sum()
            l_all = l_main+l_ssl
            aver_loss_ssl += l_ssl.cpu().item()
        else:
            l_all = l_main
        optimizer.zero_grad()
        l_all.backward()
        optimizer.step()

        aver_loss_main += l_main.cpu().item()
        aver_loss += l_all.cpu().item()
        if world.config['if_tensorboard']:
            w.add_scalar(f"BPR_Contrast_Loss/{world.config['dataset']}", l_all, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / (total_batch*batch_size)
    aver_loss_main = aver_loss_main / (total_batch*batch_size)
    aver_loss_ssl = aver_loss_ssl / (total_batch*batch_size)
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f} = {aver_loss_ssl:.3f}+{aver_loss_main:.3f}-{time_info}"

    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    #================Pop=================#
    groundTrue_popDict = X[2]#{0: [ [items of u1], [items of u2] ] }
    r, r_popDict = utils.getLabel(groundTrue, groundTrue_popDict, sorted_items)
    #================Pop=================#
    pre, recall, recall_pop, recall_pop_Contribute, ndcg = [], [], {}, {}, []
    num_group = world.config['pop_group']
    for group in range(num_group):
            recall_pop[group] = []
    for group in range(num_group):
            recall_pop_Contribute[group] = []

    for k in world.config['topks']:
        ret = utils.RecallPrecision_ATk(groundTrue, groundTrue_popDict, r, r_popDict, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])

        num_group = world.config['pop_group']
        for group in range(num_group):
            recall_pop[group].append(ret['recall_popDIct'][group])
        for group in range(num_group):
            recall_pop_Contribute[group].append(ret['recall_Contribute_popDict'][group])

        for group in range(num_group):
            recall_pop[group] = np.array(recall_pop[group])
        for group in range(num_group):
            recall_pop_Contribute[group] = np.array(recall_pop_Contribute[group])

        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'recall_popDict':recall_pop,
            'recall_Contribute_popDict':recall_pop_Contribute,
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    testDict: dict = dataset.testDict
    #================Pop=================#
    testDict_pop = dataset.testDict_pop
    #================Pop=================#
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.config['topks'])
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.config['topks'])),
               'recall': np.zeros(len(world.config['topks'])),
               'recall_pop': {},
               'recall_pop_Contribute': {},
               'ndcg': np.zeros(len(world.config['topks']))}
    num_group = world.config['pop_group']
    for group in range(num_group):
        results['recall_pop'][group] = np.zeros(len(world.config['topks']))
        results['recall_pop_Contribute'][group] = np.zeros(len(world.config['topks']))

    with torch.no_grad():
        #================Pop=================#
        RatingsPopDict = Recmodel.getItemRating()
        #================Pop=================#
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        groundTrue_list_pop = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            #================Pop=================#
            groundTrue_pop = {}
            for group, ground in testDict_pop.items():
                groundTrue_pop[group] = [ground[u] for u in batch_users]
            #================Pop=================#
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            #================Pop=================#
            groundTrue_list_pop.append(groundTrue_pop)
            #================Pop=================#
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list, groundTrue_list_pop)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
            
        for result in pre_results:
            results['recall'] += result['recall']
            for group in range(num_group):
                results['recall_pop'][group] += result['recall_popDict'][group]
                results['recall_pop_Contribute'][group] += result['recall_Contribute_popDict'][group]
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        for group in range(num_group):
            results['recall_pop'][group] /= float(len(users))
            results['recall_pop_Contribute'][group] /= float(len(users))

        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.config['if_tensorboard']:
            w.add_scalars(f"Test/Recall@{world.config['topks']}", {'@'+str(world.config['topks'][i]): results['recall'][i] for i in range(len(world.config['topks']))}, epoch)
            
            for group in range(num_group):
                w.add_scalars(f"Test-Groups/Recall_pop@{world.config['topks']}/group-{group}", {'@'+str(world.config['topks'][i]): results['recall_pop'][group][i] for i in range(len(world.config['topks']))}, epoch)
                w.add_scalars(f"Test-Groups/Recall_pop_Contribute@{world.config['topks']}/group-{group}", {'@'+str(world.config['topks'][i]): results['recall_pop_Contribute'][group][i] for i in range(len(world.config['topks']))}, epoch)
            w.add_scalars(f"Test/PopRating",  {str(group):value for group, value in RatingsPopDict.items()}, epoch)
            w.add_scalars(f"Test/Precision@{world.config['topks']}", {'@'+str(world.config['topks'][i]): results['precision'][i] for i in range(len(world.config['topks']))}, epoch)
            w.add_scalars(f"Test/NDCG@{world.config['topks']}", {'@'+str(world.config['topks'][i]): results['ndcg'][i] for i in range(len(world.config['topks']))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results

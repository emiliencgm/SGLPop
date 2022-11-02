'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go SGL")
    parser.add_argument('--temp_tau', type=float, default=0.2, help="tau in InfoNCEloss")
    parser.add_argument('--edge_drop_prob', type=float, default=0.1, help="prob to dropout egdes")
    parser.add_argument('--latent_dim_rec', type=int, default=64, help="latent dim for rec")
    parser.add_argument('--num_layers', type=int, default=3, help="num layers of LightGCN")
    parser.add_argument('--if_pretrain', type=int, default=0, help="whether use pretrained Embedding")
    parser.add_argument('--dataset', type=str, default='gowalla', help="dataset:[yelp2018,  amazon-book,  MIND]")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay == lambda2")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--model', type=str, default='SGL-ED', help="model: SGL-ED, SGL_RW")
    parser.add_argument('--if_load_embedding', type=int, default=0, help="whether load trained embedding")
    parser.add_argument('--if_tensorboard', type=int, default=1, help="whether use tensorboardX")
    parser.add_argument('--epochs', type=int, default=1000, help="training epochs")
    parser.add_argument('--if_multicore', type=int, default=1, help="whether use multicores in Test")
    parser.add_argument('--early_stop_steps', type=int, default=20, help="early stop steps")
    parser.add_argument('--bpr_batch_size', type=int, default=2048, help="batch size in BPR_Contrast_Train")
    parser.add_argument('--lambda1', type=float, default=0.1, help="lambda1 == coef of InfoNCEloss")
    parser.add_argument('--topks', nargs='?', default=[20], help="topks [@20] for test")
    parser.add_argument('--test_u_batch_size', type=int, default=2048, help="users batch size for test")
    parser.add_argument('--pop_group', type=int, default=10, help="Num of groups of Popularity")
    parser.add_argument('--if_pop', type=int, default=1, help="whether user popular-aware augmentation")
    parser.add_argument('--pop_mode', type=str, default='SGL4', help="how to augmente different popular items: ['SGL1', 'SGL2', 'GraphCL1']")
    parser.add_argument('--if_big_matrix', type=int, default=0, help="whether the adj matrix is big, and then use matrix n_fold split")
    parser.add_argument('--n_fold', type=int, default=2, help="split the matrix to n_fold")
    parser.add_argument('--cuda', type=str, default='0', help="cuda id")
    parser.add_argument('--P_e_drop1', type=float, default=0.3, help="P_e of Item Augmentation Control Hyperparameter for view 1")
    parser.add_argument('--P_e_add1', type=float, default=0.8, help="P_e of Item Augmentation to Add Edge for Cold Items for view 1")
    parser.add_argument('--P_e_drop2', type=float, default=0.3, help="P_e of Item Augmentation Control Hyperparameter for view 2")
    parser.add_argument('--P_e_add2', type=float, default=0.8, help="P_e of Item Augmentation to Add Edge for Cold Items for view 2")
    parser.add_argument('--comment', type=str, default='', help="comment for the experiment")

    return parser.parse_args()

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
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset:[yelp2018,  amazon-book,  MIND]")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay == lambda2")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
    parser.add_argument('--model', type=str, default='SGL-ED', help="Now available:\n\
                                                                     ###SGL-ED: Edge Drop (Default drop prob = edge_drop_prob = 0.1 if if_pop==0)\n\
                                                                     ###SGL-RW: Random Walk (num_layers * [sub-EdgeDrop])\n\
                                                                     Aug prob in 2 contrast (sub-)views: (P_e_drop1, P_e_add1) <--> (P_e_drop2, P_e_add2)\n\
                                                                     ###SimGCL: Strong and Simple Non Augmentation Contrastive Model")

    parser.add_argument('--if_load_embedding', type=int, default=0, help="whether load trained embedding")
    parser.add_argument('--if_tensorboard', type=int, default=1, help="whether use tensorboardX")
    parser.add_argument('--epochs', type=int, default=1000, help="training epochs")
    parser.add_argument('--if_multicore', type=int, default=1, help="whether use multicores in Test")
    parser.add_argument('--early_stop_steps', type=int, default=20, help="early stop steps")
    parser.add_argument('--bpr_batch_size', type=int, default=2048, help="batch size in BPR_Contrast_Train")
    parser.add_argument('--lambda1', type=float, default=0.1, help="lambda1 == coef of InfoNCEloss")
    parser.add_argument('--topks', nargs='?', default='[20]', help="topks [@20] for test")
    parser.add_argument('--test_u_batch_size', type=int, default=2048, help="users batch size for test")
    parser.add_argument('--pop_group', type=int, default=10, help="Num of groups of Popularity")
    parser.add_argument('--if_pop', type=int, default=1, help="whether user popular-aware augmentation")
    parser.add_argument('--pop_mode', type=str, default='SGL5', help="#SGL1\SGL2: Deleted (only for 2 pop groups)\n\
                                                                      #SGL3: Abondoned (old implementation)\n\
                                                                      Below is now available for 10 pop groups:\n\
                                                                      ###SGL4: Hot --> low drop prob       Cold --> high add prob\n\
                                                                      ###SGL5: Hot --> high drop prob      Cold --> high add prob\n\
                                                                      #SGL6: set P_e to 0. in one view (SGL4 or SGL5 contrast with Original graph, e.g. denoise Hot and diversify Cold)\n")
    #SGL4 and SGL5 are two ideas to calculate augment probs, SGL6 is just contrast one View generated by SGL4 or SGL5 with the Origin Graph.
    #For SGL6, need to set P_e_drop2 = P_e_add2 = 0. and select pop_mode = SGL4 or SGL5.
    parser.add_argument('--if_big_matrix', type=int, default=0, help="whether the adj matrix is big, and then use matrix n_fold split")
    parser.add_argument('--n_fold', type=int, default=2, help="split the matrix to n_fold")
    parser.add_argument('--cuda', type=str, default='0', help="cuda id")
    parser.add_argument('--P_e_drop1', type=float, default=0.3, help="P_e of Item Augmentation to Drop Edge for Hot Items for view 1")
    parser.add_argument('--P_e_add1', type=float, default=0.8, help="P_e of Item Augmentation to Add Edge for Cold Items for view 1")
    parser.add_argument('--P_e_drop2', type=float, default=0.3, help="P_e of Item Augmentation to Drop Edge for Hot Items for view 2")
    parser.add_argument('--P_e_add2', type=float, default=0.8, help="P_e of Item Augmentation to Add Edge for Cold Items for view 2")
    parser.add_argument('--comment', type=str, default='', help="comment for the experiment")
    parser.add_argument('--perplexity', type=int, default=50, help="perplexity for T-SNE")
    parser.add_argument('--tsne_epoch', type=int, default=1, help="t-sne visualize every tsne_epoch")
    parser.add_argument('--if_double_label', type=int, default=0, help="whether use item categories label along with popularity group")
    parser.add_argument('--if_tsne', type=int, default=1, help="whether use t-SNE")
    parser.add_argument('--tsne_group', nargs='?', default='[0, 9]', help="groups [0, 9] for t-SNE")
    parser.add_argument('--eps_SimGCL', type=float, default=0.1, help="epsilon for noise coef in SimGCL")
    parser.add_argument('--init_method', type=str, default='Xavier', help="UI embeddings init method: Xavier or Normal")
    parser.add_argument('--tsne_points', type=int, default=2000, help="Num of points of users/items in t-SNE")

    return parser.parse_args()


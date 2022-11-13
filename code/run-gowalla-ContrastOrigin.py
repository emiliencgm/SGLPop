import os
cuda = 1

os.system('python main.py --if_big_matrix 0 --n_fold 2 --edge_drop_prob 0.1 --dataset gowalla --if_pop 0 --pop_mode SGL5 --cuda {} --model SGL-ED --comment SGL6_Origin-ED'.format(cuda))
#调整drop和add的概率超参数——SGL4/5 contrast Origin Graph
for drop in [0.05, 0.1, 0.2, 0.3, 0.4]:
    for add in [0.6, 0.7, 0.8, 0.9, 1.0]:
        P_e_drop1, P_e_drop2 = drop, 0.
        P_e_add1, P_e_add2 = add, 0.
        os.system('python main.py --if_big_matrix 0 --n_fold 2 --edge_drop_prob 0.1 --dataset gowalla --if_pop 1 --pop_mode SGL4 --cuda {} --P_e_drop1 {} --P_e_add1 {} --P_e_drop2 {} --P_e_add2 {} --model SGL-ED --comment SGL6_OriginGraph_vs_SGL4-ED_HPS'.format(cuda, P_e_drop1, P_e_add1, P_e_drop2, P_e_add2))
        os.system('python main.py --if_big_matrix 0 --n_fold 2 --edge_drop_prob 0.1 --dataset gowalla --if_pop 1 --pop_mode SGL5 --cuda {} --P_e_drop1 {} --P_e_add1 {} --P_e_drop2 {} --P_e_add2 {} --model SGL-ED --comment SGL6_OriginGraph_vs_SGL5-ED_HPS'.format(cuda, P_e_drop1, P_e_add1, P_e_drop2, P_e_add2))

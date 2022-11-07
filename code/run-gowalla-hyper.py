import os
cuda = 0
#调整drop和add的概率超参数
for drop in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    for add in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
        P_e_drop1, P_e_drop2 = drop, drop
        P_e_add1, P_e_add2 = add, add
        os.system('python main.py --if_big_matrix 0 --n_fold 2 --edge_drop_prob 0.1 --dataset gowalla --if_pop 1 --cuda {} --P_e_drop1 {} --P_e_add1 {} --P_e_drop2 {} --P_e_add2 {} --comment SGL_Adaptive_HyperParamSearch_Fine'.format(cuda, P_e_drop1, P_e_add1, P_e_drop2, P_e_add2))

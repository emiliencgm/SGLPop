import os
cuda = 0
#os.system('python main.py --if_big_matrix 0 --n_fold 2 --edge_drop_prob 0.1 --dataset amazon-book --if_pop 0 --cuda {} --comment SGL_Origin'.format(cuda))
#os.system('python main.py --if_big_matrix 0 --n_fold 2 --edge_drop_prob 0.1 --dataset amazon-book --if_pop 1 --cuda {} --comment SGL_Adaptive'.format(cuda))
#os.system('python main.py --if_big_matrix 0 --n_fold 2 --edge_drop_prob 0.0 --dataset amazon-book --if_pop 0 --cuda {} --lambda1 0. --comment LightGCN'.format(cuda))
#os.system('python main.py --if_big_matrix 0 --n_fold 2 --edge_drop_prob 0.0 --dataset amazon-book --if_pop 0 --cuda {} --comment LGN+InfoNCE'.format(cuda))

#调整drop和add的概率超参数
for drop in [0.1, 0.2, 0.3]:
    for add in [0.8, 0.9, 1.0]:
        if add == 0.8 and drop==0.1:
            continue
        P_e_drop1, P_e_drop2 = drop, drop
        P_e_add1, P_e_add2 = add, add
        os.system('python main.py --if_big_matrix 0 --n_fold 1 --edge_drop_prob 0.1 --dataset amazon-book --if_pop 1 --cuda {} --P_e_drop1 {} --P_e_add1 {} --P_e_drop2 {} --P_e_add2 {} --comment SGL_Adaptive_HyperParamSearch'.format(cuda, P_e_drop1, P_e_add1, P_e_drop2, P_e_add2))


os.system('python main.py --if_big_matrix 0 --n_fold 1 --edge_drop_prob 0.1 --dataset amazon-book --if_pop 0 --cuda {} --model SGL-RW --comment SGL_Origin-RW'.format(cuda))
#调整drop和add的概率超参数
for drop in [0.05, 0.1, 0.2, 0.3, 0.4]:
    for add in [0.7, 0.8, 0.9, 1.0]:
        P_e_drop1, P_e_drop2 = drop, drop
        P_e_add1, P_e_add2 = add, add
        os.system('python main.py --if_big_matrix 0 --n_fold 1 --edge_drop_prob 0.1 --dataset amazon-book --if_pop 1 --cuda {} --P_e_drop1 {} --P_e_add1 {} --P_e_drop2 {} --P_e_add2 {} --model SGL-RW --comment SGL_Adaptive_RW_HyperParamSearch'.format(cuda, P_e_drop1, P_e_add1, P_e_drop2, P_e_add2))
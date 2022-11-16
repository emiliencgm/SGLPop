import os
#LightGCN
os.system("python main.py --cuda 1 --dataset yelp2018 --model SGL-ED --if_pop 0 --pop_mode SGL5 --edge_drop_prob 0.0 --lambda1 0.0 --P_e_drop1 0.1 --P_e_add1 0.8 --P_e_drop2 0.0 --P_e_add2 0.0 --temp_tau 0.2 --init_method Xavier --topks [10,20,50] --comment LightGCN")
#LightGCN + InfoNCE
os.system("python main.py --cuda 1 --dataset yelp2018 --model SGL-ED --if_pop 0 --pop_mode SGL5 --edge_drop_prob 0.0 --lambda1 0.1 --P_e_drop1 0.1 --P_e_add1 0.8 --P_e_drop2 0.0 --P_e_add2 0.0 --temp_tau 0.2 --init_method Xavier --topks [10,20,50] --comment LightGCN+InfoNCE")
#SimGCL
os.system("python main.py --cuda 1 --dataset yelp2018 --model SimGCL --if_pop 0 --pop_mode SGL5 --edge_drop_prob 0.0 --lambda1 0.0 --P_e_drop1 0.1 --P_e_add1 0.8 --P_e_drop2 0.0 --P_e_add2 0.0 --temp_tau 0.2 --init_method Xavier --topks [10,20,50] --comment SimGCL")
#SGL-ED-Origin
os.system("python main.py --cuda 1 --dataset yelp2018 --model SGL-ED --if_pop 0 --pop_mode SGL5 --edge_drop_prob 0.1 --lambda1 0.1 --P_e_drop1 0.1 --P_e_add1 0.8 --P_e_drop2 0.0 --P_e_add2 0.0 --temp_tau 0.2 --init_method Xavier --topks [10,20,50] --comment SGL-ED-Origin")
#SGL-RW-Origin
os.system("python main.py --cuda 1 --dataset yelp2018 --model SGL-RW --if_pop 0 --pop_mode SGL5 --edge_drop_prob 0.1 --lambda1 0.1 --P_e_drop1 0.1 --P_e_add1 0.8 --P_e_drop2 0.0 --P_e_add2 0.0 --temp_tau 0.2 --init_method Xavier --topks [10,20,50] --comment SGL-RW-Origin")
#SGL6-ED: SGL5-ED vs Origin Graph
os.system("python main.py --cuda 1 --dataset yelp2018 --model SGL-ED --if_pop 1 --pop_mode SGL5 --edge_drop_prob 0.1 --lambda1 0.1 --P_e_drop1 0.1 --P_e_add1 0.8 --P_e_drop2 0.0 --P_e_add2 0.0 --temp_tau 0.2 --init_method Xavier --topks [10,20,50] --comment SGL6-ED-Adaptive")
#SGL5-ED: Adaptive-ED
os.system("python main.py --cuda 1 --dataset yelp2018 --model SGL-ED --if_pop 1 --pop_mode SGL5 --edge_drop_prob 0.1 --lambda1 0.1 --P_e_drop1 0.1 --P_e_add1 0.8 --P_e_drop2 0.1 --P_e_add2 0.8 --temp_tau 0.2 --init_method Xavier --topks [10,20,50] --comment SGL5-ED-Adaptive")
#perplexity & tsne_points
'''
--perplexity 50
--tsne_points 2000 
'''
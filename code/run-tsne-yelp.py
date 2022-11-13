import os
cuda = 1
os.system('python main.py --edge_drop_prob 0.0 --dataset yelp2018 --if_pop 1 --pop_mode SGL5 --cuda {} --P_e_drop1 0.4, P_e_add1 0.7, P_edrop2 0.0, P_e_add2 0.0 --comment SGL6(BestRecall@14)'.format(cuda))
os.system('python main.py --edge_drop_prob 0.0 --dataset yelp2018 --if_pop 1 --pop_mode SGL5 --cuda {} --P_e_drop1 0.3, P_e_add1 0.8, P_edrop2 0.0, P_e_add2 0.0 --comment SGL6(BestRecall@16)'.format(cuda))
os.system('python main.py --edge_drop_prob 0.0 --dataset yelp2018 --if_pop 1 --pop_mode SGL4 --cuda {} --P_e_drop1 0.2, P_e_add1 1.0, P_edrop2 0.2, P_e_add2 1.0 --comment SGL4(BestCold@14-16)'.format(cuda))
os.system('python main.py --edge_drop_prob 0.0 --dataset yelp2018 --if_pop 1 --pop_mode SGL5 --cuda {} --P_e_drop1 0.3, P_e_add1 1.0, P_edrop2 0.3, P_e_add2 1.0 --comment SGL5(BestHot@16)'.format(cuda))
os.system('python main.py --edge_drop_prob 0.0 --dataset yelp2018 --if_pop 1 --pop_mode SGL5 --cuda {} --P_e_drop1 0.05, P_e_add1 0.8, P_edrop2 0.05, P_e_add2 0.8 --comment SGL4(BestHot@15)'.format(cuda))
os.system('python main.py --edge_drop_prob 0.0 --dataset yelp2018 --if_pop 1 --pop_mode SGL4 --cuda {} --P_e_drop1 0.3, P_e_add1 0.8, P_edrop2 0.3, P_e_add2 0.8 --comment SGL4(BestHot@14)'.format(cuda))
os.system('python main.py --edge_drop_prob 0.0 --dataset yelp2018 --if_pop 0 --pop_mode SGL5 --cuda {} --P_e_drop1 0.0, P_e_add1 0.0, P_edrop2 0.0, P_e_add2 0.0 -- lambda1 0.0 --comment LightGCN'.format(cuda))
os.system('python main.py --edge_drop_prob 0.0 --dataset yelp2018 --if_pop 0 --pop_mode SGL5 --cuda {} --P_e_drop1 0.0, P_e_add1 0.0, P_edrop2 0.0, P_e_add2 0.0 --comment LightGCN+InfoNCE'.format(cuda))
os.system('python main.py --edge_drop_prob 0.1 --dataset yelp2018 --if_pop 0 --pop_mode SGL5 --cuda {} --P_e_drop1 0.0, P_e_add1 0.0, P_edrop2 0.0, P_e_add2 0.0 --comment SGL_Origin'.format(cuda))








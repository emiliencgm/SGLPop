'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config = {}
config['temp_tau'] = args.temp_tau
config['edge_drop_prob'] = args.edge_drop_prob
config['latent_dim_rec'] = args.latent_dim_rec
config['num_layers'] = args.num_layers
config['if_pretrain'] = args.if_pretrain
config['dataset'] = args.dataset
config['lr'] = args.lr
config["weight_decay"] = args.weight_decay
config['seed'] = args.seed
config['model'] = args.model
config['if_load_embedding'] = args.if_load_embedding
config['if_tensorboard'] = args.if_tensorboard
config['epochs'] = args.epochs
config['if_multicore'] = args.if_multicore
config['early_stop_steps'] = args.early_stop_steps
config['bpr_batch_size'] = args.bpr_batch_size
config['lambda1'] = args.lambda1
config['topks'] = args.topks
config['test_u_batch_size'] = args.test_u_batch_size
config['pop_group'] = args.pop_group
config['if_pop'] = args.if_pop
config['pop_mode'] = args.pop_mode
config['if_big_matrix'] = args.if_big_matrix
config['n_fold'] = args.n_fold
config['P_e_drop1'] = args.P_e_drop1
config['P_e_add1'] = args.P_e_add1
config['P_e_drop2'] = args.P_e_drop2
config['P_e_add2'] = args.P_e_add2
#备注
config['comment'] = args.comment
#加载预训练的embedding
config['user_emb'] = 0
config['item_emb'] = 0

#打印在TensorboardX的参数信息
log = {}
#LogItems = ['temp_tau', 'edge_drop_prob', 'num_layers', 'lr', 'weight_decay', 'seed', 'bpr_batch_size', 'lambda1', 'dataset', 'pop_rate', 'if_pop', 'pop_mode']
LogItems = ['if_pop', 'pop_mode', 'edge_drop_prob', 'lambda1', 'P_e_drop1', 'P_e_add1', 'P_e_drop2', 'P_e_add2', 'comment']
for key in LogItems:
    log[key] = config[key]


#TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!记得修改根目录
ROOT_PATH = "/home/newdisk/cgm/code/Pop2"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, f"runs/{config['dataset']}")
FILE_PATH = join(CODE_PATH, 'checkpoints')
LOG_FILE = join(CODE_PATH, 'Log')

import sys
sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)
if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE, exist_ok=True)


GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
config['device'] = device
print('DEVICE:',device)
#print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.get_device_name(device))

CORES = multiprocessing.cpu_count() // 2
config['cores'] = CORES
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

def cprint(words : str):
    #print(words)
    print(f"\033[0;30;43m{words}\033[0m")




def make_print_to_file(path=LOG_FILE):
    import sys
    import os
    import config_file as cfg_file
    import sys
    import datetime
 
    class Logger(object):
        def __init__(self, filename="Default.log", path=path):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
 
    fileName = datetime.datetime.now().strftime('%Y_%m_%d')
    sys.stdout = Logger(fileName + f"-{config['dataset']}-" +'.log', path=path)
 
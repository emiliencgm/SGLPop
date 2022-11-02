from torch_sparse.tensor import to
from tqdm import tqdm
from contrast import Contrast
from torch import optim
from torch.optim import optimizer, lr_scheduler
import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
world.make_print_to_file()
# ==============================
utils.set_seed(world.config['seed'])
print(">>SEED:", world.config['seed'])
# ==============================
import register
from register import dataset
Recmodel = register.MODELS[world.config['model']](world.config, dataset)
Recmodel = Recmodel.to(world.device)
contrast_model = Contrast(Recmodel).to(world.device)
optimizer = optim.Adam(Recmodel.parameters(), lr=world.config['lr'])
if world.config['dataset'] == "MIND":
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.2)#学习率在5和10轮后分别变为之前的0.2倍
else:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1500, 2500], gamma = 0.2)
bpr = utils.BPRLoss(Recmodel, optimizer)
weight_file = utils.getFileName()
print(f'load and save to {weight_file}')
if world.config['if_load_embedding']:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.config['if_tensorboard']:
    w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + str([(key,value)for key,value in world.log.items()])))
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    # for early stop
    # recall@20
    least_loss = 1e5
    best_result = 0.
    stopping_step = 0

    for epoch in tqdm(range(world.config['epochs']), disable=True):
        start = time.time()
        cprint("[Data Augment]")
        contrast_views = contrast_model.get_views()
        cprint("[Joint Training]")
        output_information = Procedure.BPR_train_contrast(dataset, Recmodel, bpr, contrast_model, contrast_views, epoch, optimizer, neg_k=Neg_k,w=w)            
        print(f"EPOCH[{epoch+1}/{world.config['epochs']}] {output_information}")

        if epoch % 1== 0:
            cprint("[TEST]")
            result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['if_multicore'])
            if result["recall"] > best_result:
                stopping_step = 0
                best_result = result["recall"]
                print("find a better model")
                torch.save(Recmodel.state_dict(), weight_file)
            else:
                stopping_step += 1
                if stopping_step >= world.config['early_stop_steps']:
                    print(f"early stop triggerd at epoch {epoch}")
                    break
        
        scheduler.step()
finally:
    if world.config['if_tensorboard']:
        w.close()
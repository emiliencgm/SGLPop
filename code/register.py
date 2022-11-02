import world
import dataloader
import model
import utils
from pprint import pprint
from os.path import join
import os
from pprint import pprint

if world.config['dataset'] in ['gowalla', 'MIND', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path=join(world.DATA_PATH,world.config['dataset']))

print('==========config==========')
pprint(world.config)
print('==========config==========')


MODELS = {
    'SGL-ED': model.SGL_ED,
    'SGL-RW': model.SGL_ED
}
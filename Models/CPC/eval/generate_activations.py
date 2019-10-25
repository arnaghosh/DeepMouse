import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tmodels
from functools import partial
import collections

curr_wd = os.getcwd()
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
print(os.getcwd())
sys.path.append('../backbone/')
sys.path.append('../dpc/')
print(sys.path)
os.chdir(curr_wd)
from resnet_2d3d import neq_load_customized
from model_3d import *


'''

TO DO: 
- add dataloader and transform for the data
- add option for choosing layers

'''

def get_activations(PATH,batch):
	
    '''
    
	PATH: path to a saved pretrained model
	batch: numpy array of images/stimuli of size (batch size X number of blocks X colors X number of frames X height X width)
	Output: a dictionary containing layer activations as tensors
    
	'''
    model = DPC_RNN(sample_size=48, 
                        num_seq=8, 
                        seq_len=5, 
                        network='resnet18', 
                        pred_step=3)

    checkpoint = torch.load(PATH)
    model = neq_load_customized(model, checkpoint['state_dict'])

    activations = collections.defaultdict(list)
	
    for name, m in model.named_modules():
    		if type(m)==nn.Conv3d:
        		print(name)
        		# partial to assign the layer name to each hook
        		m.register_forward_hook(partial(save_activation, name))

    for batch in dataset:
        out = model(batch)

    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    return activations


def save_activation(name, mod, inp, out):
	activations[name].append(out.cpu())	


if __name__=="__main__":
	
	# path to sample pretrained saved model
	PATH = "../pretrained/testmodel.pth.tar"
	
	# dummy dataset: 10 batches of size 5
	batch = [torch.rand(5,8,3,5,56,56) for _ in range(10)] # B x N x C x T x W x H
	
	activations = get_activations(PATH,batch)




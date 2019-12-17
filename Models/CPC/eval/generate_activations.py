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
sys.path.append(os.path.join(os.getcwd(),'../backbone/'))
sys.path.append(os.path.join(os.getcwd(),'../dpc/'))
os.chdir(curr_wd)
from resnet_2d3d import neq_load_customized
from model_3d import *


'''

TO DO: 
- add dataloader and transform for the data
- add option for choosing layers

'''

def get_activations(PATH,dataset):
	
    '''
    
	PATH: path to a saved pretrained model
	batch: numpy array of images/stimuli of size (batch size X number of blocks X colors X number of frames X height X width)
	Output: a dictionary containing layer activations as tensors
    
	'''
    model = DPC_RNN(sample_size=184,#48, 
                        num_seq=118,#8 
                        seq_len=5, 
                        network='resnet18', 
                        pred_step=3)

    checkpoint = torch.load(PATH,map_location=torch.device('cpu'))
    model = neq_load_customized(model, checkpoint['state_dict'])

    activations = collections.defaultdict(list)
	
    for name, m in model.named_modules():
    		if type(m)==nn.Conv3d:
        		print(name)
        		# partial to assign the layer name to each hook
        		m.register_forward_hook(partial(save_activation, activations, name))
    with torch.no_grad():
        for batch in dataset:
            out = model(batch)

        activations = {name: torch.cat(outputs, 0).detach() for name, outputs in activations.items()}
    for key,value in activations.items():
        activations[key] = value.detach().numpy()
    return activations


def save_activation(activations, name, mod, inp, out):#save_activation(name, mod, inp, out,activations):
    activations[name].append(out.cpu().detach())	
    

if __name__=="__main__":
	
	# path to sample pretrained saved model
	PATH = "../pretrained/testmodel.pth.tar"
	
	# dummy dataset: 10 batches of size 5
	dataset = [torch.rand(5,8,3,5,56,56) for _ in range(10)] # B x N x C x T x W x H
	
	activations = get_activations(PATH,dataset);


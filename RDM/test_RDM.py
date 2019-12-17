import os, sys
from generate_SSM import *
curr_wd = os.getcwd()
sys.path.append(os.path.join(os.getcwd(),'../'))


from Models.CPC.eval.generate_activations import *

PATH = os.path.join(os.getcwd(),"../Models/CPC/pretrained/testmodel.pth.tar")
dataset = [torch.rand(5,8,3,5,56,56) for _ in range(10)] # B x N x C x T x W x H


activations = get_activations(PATH,dataset)

sim_mats = compute_similarity_matrices(activations)
print("SSM values between first 2 entries:",compute_ssm(sim_mats[list(sim_mats.keys())[0]],sim_mats[list(sim_mats.keys())[1]]))
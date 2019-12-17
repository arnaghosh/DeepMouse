import os, sys
from generate_SSM import *
curr_wd = os.getcwd()
sys.path.append(os.path.join(os.getcwd(),'../'))
sys.path.append('./boc/')
from Models.CPC.eval.generate_activations import *
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# %matplotlib inline 
import cv2
import scipy as sp
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tmodels
from functools import partial
import collections



# PATH_CPC = os.path.join(os.getcwd(),"../Models/CPC/pretrained/testmodel.pth.tar")
PATH_CPC = os.path.join(os.getcwd(),"../../../Results/DeepMouse/TrainedModel/log_tmp/ucf101-128_r18_dpc-rnn_bs10_lr0.001_seq8_pred3_len5_ds3_train-all/model/model_best_epoch88.pth.tar")
PATH_visualStim = os.path.join(os.getcwd(),"../../Visual-Stimulus/StimulusImages/")

def prePareVisualStim_for_CPC(blocksize):
    
    allDGImage_paths = glob.glob(os.path.join(PATH_visualStim,"DG/*.png"))
    numDGImages = len(allDGImage_paths)
    numBlocks = int(numDGImages/blocksize)
    data = np.ndarray((1,numBlocks,1,blocksize,184,184))#scenes.shape[1],scenes.shape[1]
    f = 0
    for n in range(0,numBlocks):
        img = cv2.imread(allDGImage_paths[f],0)
        for b in range(0,blocksize):
            thisImage = np.array(img)
            thisImage = cv2.resize(thisImage,(184,184))
#             thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)

            data[0,n,0,b,:,:] = thisImage
            f = f + 1
    data_DG = np.concatenate((data,data,data),axis=2)
    
    
    allSGImage_paths = glob.glob(os.path.join(PATH_visualStim,"SG/*.png"))
    numSGImages = len(allSGImage_paths)
    data = np.ndarray((1,numSGImages,1,blocksize,184,184))#scenes.shape[1],scenes.shape[1]

    for n in range(0,numSGImages):
        img = cv2.imread(allSGImage_paths[n],0)
        for b in range(0,blocksize):
            thisImage = np.array(img)
            thisImage = cv2.resize(thisImage,(184,184))
#             thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)

            data[0,n,0,b,:,:] = thisImage

    data_SG = np.concatenate((data,data,data),axis=2)
    
    allRDKImage_paths = glob.glob(os.path.join(PATH_visualStim,"RDK/*.png"))
    numRDKImages = len(allRDKImage_paths)
    numBlocks = int(numRDKImages/blocksize)
    data = np.ndarray((1,numBlocks,1,blocksize,184,184))#scenes.shape[1],scenes.shape[1]
    f = 0
    for n in range(0,numBlocks):
        img = cv2.imread(allRDKImage_paths[f],0)
        for b in range(0,blocksize):
            thisImage = np.array(img)
            thisImage = cv2.resize(thisImage,(184,184))
#             thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)

            data[0,n,0,b,:,:] = thisImage
            f = f + 1
    data_RDK = np.concatenate((data,data,data),axis=2)

    return data_DG, data_SG, data_RDK

def prePareAllenStim_for_CPC(exp_id,blocksize):
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(exp_id)
    
    scenes = data_set.get_stimulus_template('natural_scenes')
    movie = data_set.get_stimulus_template('natural_movie_one')
    
    numImages = scenes.shape[0]
    data = np.ndarray((1,numImages,1,blocksize,184,184))#scenes.shape[1],scenes.shape[1]
    for n in range(0,numImages):
        for b in range(0,blocksize):
            thisImage = np.array(scenes[n,:,0:918])
            thisImage = cv2.resize(thisImage,(184,184))
            thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)

            data[0,n,0,b,:,:] = thisImage
    
    data_colored = np.concatenate((data,data,data),axis=2)
    
    numFrames = movie.shape[0]
    numBlocks = int(numFrames/(3*blocksize))
    data = np.ndarray((1,numBlocks,1,blocksize,184,184))#scenes.shape[1],scenes.shape[1]
    f = 0
    nidx = np.arange(numBlocks)
    for n in range(0,numBlocks):
        for b in range(0,blocksize):
            
            thisImage = np.array(movie[f,:,0:303])
            thisImage = cv2.resize(thisImage,(184,184))
            thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)

            data[0,nidx[n],0,b,:,:] = thisImage
            f = f + 3
    
    data2_colored = np.concatenate((data,data,data),axis=2)
    
    
    return data_colored, data2_colored

def prePareVisualStim_for_othermodels():
    
    
    allSGImage_paths = glob.glob(os.path.join(PATH_visualStim,"SG/*.png"))
    numSGImages = len(allSGImage_paths)
    data = np.ndarray((numSGImages,1,224,224))#scenes.shape[1],scenes.shape[1]

    for n in range(0,numSGImages):
        img = cv2.imread(allSGImage_paths[n],0)
        thisImage = np.array(img)
        thisImage = cv2.resize(thisImage,(224,224))
#             thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)
        data[n,0,:,:] = thisImage

    data_SG = np.concatenate((data,data,data),axis=1)
    print(data_SG.shape)

    return data_SG

def prePareAllenStim_for_othermodels(exp_id):
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(exp_id)
    
    scenes = data_set.get_stimulus_template('natural_scenes')
    movie = data_set.get_stimulus_template('natural_movie_one')
    
    numImages = scenes.shape[0]
    data = np.ndarray((numImages,1,224,224))#scenes.shape[1],scenes.shape[1]
    for n in range(0,numImages):
        thisImage = np.array(scenes[n,:,0:918])
        thisImage = cv2.resize(thisImage,(224,224))
        thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)
        data[n,0,:,:] = thisImage
    
    data_colored = np.concatenate((data,data,data),axis=1)
    return data_colored

def get_activations_othermodels(data_,ModelName):
    
    if ModelName == 'alexnet':
        net = tmodels.alexnet(pretrained=True)
    elif ModelName == 'vgg16':
        net = tmodels.vgg16(pretrained=True)
        

    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu())

    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    for name, m in net.named_modules():
        if type(m)==nn.Conv2d:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, name))

    # forward pass through the full dataset
    for batch in data_:
        out = net(batch)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    for name in activations.keys():
        activations[name] = activations[name].detach()
    
    return activations

def get_othermodels_RSMs(StimType,ModelName):
    
    if StimType == 'static_gratings':
        data = prePareVisualStim_for_othermodels()
    
    elif StimType == 'natural_scenes':    
        data = prePareAllenStim_for_othermodels(501498760)
    
    dataset = [(torch.Tensor(data[:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
    activations = get_activations_othermodels(dataset,ModelName)
    
    all_RSM = compute_similarity_matrices(activations)
    
    return all_RSM
    
def get_CPC_RSMs(StimType):
    

    if StimType == 'drifting_gratings':
        data_DG, _, _ = prePareVisualStim_for_CPC(5)
        dataset = [(torch.Tensor(data_DG[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations = get_activations(PATH_CPC,dataset)
        all_RSM = compute_similarity_matrices(activations)
        del dataset
        
    elif StimType == 'static_gratings':
        _, data_SG, _ = prePareVisualStim_for_CPC(5)
        dataset = [(torch.Tensor(data_SG[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations = get_activations(PATH_CPC,dataset)
        all_RSM = compute_similarity_matrices(activations)
        del dataset
    
    elif StimType == 'rdk':
        _, _, data_RDK = prePareVisualStim_for_CPC(5)
        dataset = [(torch.Tensor(data_RDK[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations = get_activations(PATH_CPC,dataset)
        all_RSM = compute_similarity_matrices(activations)
        del dataset
    
    elif StimType == 'natural_scenes':
        data, _ = prePareAllenStim_for_CPC(501498760,5)
        dataset = [(torch.Tensor(data[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations = get_activations(PATH_CPC,dataset)
        all_RSM = compute_similarity_matrices(activations)
        del dataset
        
    elif StimType == 'natural_movies':
        _, data = prePareAllenStim_for_CPC(501498760,5)
        dataset = [(torch.Tensor(data[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations = get_activations(PATH_CPC,dataset)
        all_RSM = compute_similarity_matrices(activations)
        del dataset
        
    
    return all_RSM
    
    
# for i in range(len(sim_mats_images.keys())):
#     conv1 = sim_mats_images[list(sim_mats_images.keys())[i]]
# #     conv1 = conv1 - np.nanmedian(conv1)
#     np.fill_diagonal(conv1,'nan')
#     fig, ax = plt.subplots()
#     x = ax.imshow(conv1,cmap='seismic')
#     fig.colorbar(x,ax=ax)
#     plt.savefig('/Users/shahab/Mila/Results/DeepMouse/RSM/RSM_DG_'+str(i))
    


# for i in range(len(sim_mats_images.keys())):
#     conv1 = sim_mats_images[list(sim_mats_images.keys())[i]]
# #     conv1 = conv1 - np.nanmedian(conv1)
#     np.fill_diagonal(conv1,'nan')
#     fig, ax = plt.subplots()
#     x = ax.imshow(conv1,cmap='seismic')
#     fig.colorbar(x,ax=ax)
#     plt.savefig('/Users/shahab/Mila/Results/DeepMouse/RSM/RSM_SG_'+str(i))


# for i in range(len(sim_mats_images.keys())):
#     conv1 = sim_mats_images[list(sim_mats_images.keys())[i]]
# #     conv1 = conv1 - np.nanmedian(conv1)
#     np.fill_diagonal(conv1,'nan')
#     fig, ax = plt.subplots()
#     x = ax.imshow(1-conv1)
#     fig.colorbar(x,ax=ax)
#     plt.savefig('/Users/shahab/Mila/Results/DeepMouse/RSM/RSM_RDK_'+str(i))
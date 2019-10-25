import numpy as np
import torch
from tqdm import tqdm
from Stimuli import StimuliDataset
import sys, os

curr_wd = os.getcwd()
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
sys.path.append(os.path.join(os.getcwd(),'../Models/CPC/eval/'))
os.chdir(curr_wd)

from generate_activations import *
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

def generate_features(model_path, dataset, transform=None, layers=None):
	'''
	model_path: path to saved DNN to generate features from - will be used by get_activations function
	dataset: numpy array of images/stimuli of size (number of datapoints X channels/cont. frames X height X width)
	transform: torch transform to be applied to stimuli
	layers: list of model layers for which features are to be generated

	Output: a dictionary containing layer activations as numpy arrays
	'''
	dataset_stim = StimuliDataset(dataset,transform)
	loader = torch.utils.data.Dataloader(dataset_stim, num_workers=1, batch_size=1, shuffle=False)

	# pass a random input to model to retrieve the length of activations list
	inp_shape = dataset.shape[1:]	# get dimension of 1 datapoint
	rand_inp = np.random.rand(*inp_shape)
	rand_activations = get_activations(model_path,rand_inp)
	num_activations = len(rand_activations)

	feat_dict = {}
	if isinstance(rand_activations,list):	# if get_activations returns a list
		if layers:
			for l in layers:
				feat_dict[l] = []

			with torch.no_grad():
				for iter,x in enumerate(loader):
					activations = get_activations(model_path,x)
					for l in layers:
						feat_dict[l].append(activations[l])
		
		else:
			for l in range(num_activations):
				feat_dict[l] = []

			with torch.no_grad():
				for iter,x in enumerate(loader):
					activations = get_activations(model_path,x)
					for l in range(num_activations):
						feat_dict[l].append(activations[l])

	elif isinstance(rand_activations,dict):		# if get_activations returns a dict
		if layers:
			for l in layers:
				feat_dict[l] = []

			with torch.no_grad():
				for iter,x in enumerate(loader):
					activations = get_activations(model_path,x)
					for l in layers:
						assert l in activations.keys(), "Layer "+str(l) + " is not present in model!!"
						feat_dict[l].append(activations[l])
		
		else:
			for l in rand_activations.keys():
				feat_dict[l] = []

			with torch.no_grad():
				for iter,x in enumerate(loader):
					activations = get_activations(model_path,x)
					for l in rand_activations.keys():
						feat_dict[l].append(activations[l])

	else:
		raise NotImplementedError
	for k in feat_dict.keys():
		feat_dict[k] = np.array(feat_dict[k])
		print("Shape of activation array of layer ",k,": ",feat_dict[k].shape)

	return feat_dict

def compute_SVCCA(activation1, activation2):
	'''
	activation1 - Activation array 1 as a numpy array of size n X m1 
	activation2 - Activation array 2 as a numpy array of size n X m2

	'''
	pca_r = 40	# value from Shi et al NeurIPS 2019
	n = activation1.shape[0]
	assert n==activation2.shape[0], "Size of activation arrays are different!!"
	if pca_r > activation1.shape[1]:
		print("Activation 1 array has less neurons.. changing number of PCs to ",activation1.shape[1])
		pca_r = activation1.shape[1]
	if pca_r > activation2.shape[1]:
		print("Activation 2 array has less neurons.. changing number of PCs to ",activation2.shape[1])
		pca_r = activation2.shape[1]

	pca1 = PCA(n_components=pca_r)
	red_activation1 = pca1.fit_transform(activation1)
	pca2 = PCA(n_components=pca_r)
	red_activation2 = pca2.fit_transform(activation2)
	cca = CCA(n_components = pca_r)
	red_activation1_c, red_activation2_c = cca.fit_transform(red_activation1,red_activation2)
	corr_values = np.zeros(pca_r)
	for idx in range(pca_r):
		corr_values[idx] = np.corrcoef(red_activation1_c[:,idx],red_activation2_c[:,idx])[0,1]	# get the off-diagonal element

	return np.mean(corr_values)

if __name__=="__main__":
	# A placeholder function to illustrate usage of the functions
	N_stim = 100
	A = np.random.rand(N_stim,300)	# random array containing activations of 300 units
	B = np.random.rand(N_stim,500)	# random array containing activations of 500 units
	A_corrupt = 2*A + 0.01*np.random.rand(*A.shape)	# some corrupt version of 2*A
	SVCCA_12 = compute_SVCCA(A,B)
	SVCCA_12_corrupt = compute_SVCCA(A_corrupt,B)
	SVCCA_11_corrupt = compute_SVCCA(A,A_corrupt)
	print("SVCCA values for 2 layer representations :",SVCCA_12)
	print("SVCCA values for 2 layer representations with noise :",SVCCA_12_corrupt)
	print("SVCCA values for a layer representations (with and without noise):",SVCCA_11_corrupt)
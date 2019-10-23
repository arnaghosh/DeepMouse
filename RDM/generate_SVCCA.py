import numpy as np
import torch
from tqdm import tqdm
from Stimuli import StimuliDataset
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

def generate_features(model, dataset, transform=None, layers=None):
	'''
	model: DNN to generate features from - should have a method named "get_activations" that returns a list of layer activations
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
	rand_activations = model.get_activations(rand_inp)
	num_activations = len(rand_activations)

	feat_dict = {}
	if layers:
		for l in layers:
			feat_dict[l] = []

		with torch.no_grad():
			for iter,x in enumerate(loader):
				activations = model.get_activations(x)
				for l in layers:
					feat_dict[l].append(activations[l])
	
	else:
		for l in range(num_activations):
			feat_dict[l] = []

		with torch.no_grad():
			for iter,x in enumerate(loader):
				activations = model.get_activations(x)
				for l in range(num_activations):
					feat_dict[l].append(activations[l])

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


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

def compute_similarity_matrices(feature_dict, layers=None):
	'''
	feature_dict: a dictionary containing layer activations as numpy arrays
	layers: list of model layers for which features are to be generated

	Output: a dictionary containing layer activation similarity matrices as numpy arrays
	'''
	similarity_mat_dict = {}
	if layers is not None:
		for layer in layers:
			try:
				activation_arr = feature_dict[layer]
				activations_flattened = activation_arr.reshape((activation_arr.shape[0],-1))
				similarity_mat_dict[layer] = np.corrcoef(activations_flattened)
			except Exception as e:
				print(layer)
				raise e
	else:
		for layer,activation_arr in feature_dict.items():
			try:
				activations_flattened = activation_arr.reshape((activation_arr.shape[0],-1))
				similarity_mat_dict[layer] = np.corrcoef(activations_flattened)
			except Exception as e:
				print(layer,activation_arr.shape)
				raise e

	return similarity_mat_dict

def shuffle_similarity_mat(similarity_mat):
	'''
	similarity_mat: similarity matrix as a numpy array of size n X n

	Output: a random permuted order of similarity_mat (rows and columns permuted using the same order, i.e. order of stimuli changed)
	'''
	n = similarity_mat.shape[0]
	p = np.random.permutation(n)
	random_similarity_mat = similarity_mat[p]	# permute the rows
	random_similarity_mat = (random_similarity_mat.T[p]).T 	# permute the columns
	return random_similarity_mat

def compute_ssm(similarity1, similarity2, num_shuffles=None, num_folds=None):
	'''
	similarity1: first similarity matrix as a numpy array of size n X n
	similarity2: second similarity matrix as a numpy array of size n X n
	num_shuffles: Number of shuffles to perform to generate a distribution of SSM values
	num_folds: Number of folds to split stimuli set into

	Output: the spearman rank correlation of the similarity matrices
	'''
	if num_shuffles is not None:
		raise NotImplementedError()

	if num_folds is not None:
		raise NotImplementedError()

	try:
		from scipy.stats import spearmanr
		r,_ = spearmanr(similarity1.flatten(),similarity2.flatten())
		return r
	except:
		print("Error in calculating spearman correlation")
		raise

if __name__=="__main__":
	# A placeholder function to illustrate usage of the functions
	N_stim = 100
	A = np.random.rand(N_stim,300)	# random array containing activations of 300 units
	B = np.random.rand(N_stim,500)	# random array containing activations of 500 units
	A_corrupt = 2*A + 0.01*np.random.rand(*A.shape)	# some corrupt version of 2*A
	activation_dict = {'L1':A,'L1_corrupt':A_corrupt,'L2':B}
	similarity_dict = compute_similarity_matrices(activation_dict)
	ssm_12 = compute_ssm(similarity_dict['L1'],similarity_dict['L2'])
	ssm_12_corrupt = compute_ssm(similarity_dict['L1_corrupt'],similarity_dict['L2'])
	ssm_11_corrupt = compute_ssm(similarity_dict['L1'],similarity_dict['L1_corrupt'])
	print("SSM values for 2 layer representations :",ssm_12)
	print("SSM values for 2 layer representations with noise :",ssm_12_corrupt)
	print("SSM values for a layer representations (with and without noise):",ssm_11_corrupt)
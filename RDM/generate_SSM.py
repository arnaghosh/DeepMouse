import numpy as np
import torch
from tqdm import tqdm

class StimuliDataset(torch.utils.data.Dataset):
	def __init__(self,dataset,transform=None):
		'''
		dataset: numpy array of images/stimuli
		transform: transform to be applied to image
		'''
		self.images = dataset
		self.len_dataset = len(self.images)
		self.transform = transform

	def __len__(self):
		return self.len_dataset

	def __getitem__(self,index):
		img = self.images[index]
		if self.transform:
			img = self.transform(img)
		return img


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

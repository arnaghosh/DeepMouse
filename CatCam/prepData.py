# Import stuff here
import numpy as np

def prepDataArrayFromMovie(movieName, num_cont_frames, temp_downsampling, spatial_downsampling):
  '''
  movieName: Name of the folder to extract data frames from
  num_cont_frames: Number of contiguous frames to append to create a data frame
  temp_downsampling: Temporal downsampling factor, picks every t_{th} frame from folder
  spatial_downsampling: Spatial downsampling factor, reduces image size by a factor of 's'
  
  Output: a 4D numpy tensor/array containing data frames from specified movie, Size: [M_i, num_cont_frames, Height, Width]
          M_i is number of data frames from movie.
  '''
  pass
  
def  createTrainValidationSplits(mode, split_ratio, save=False):
  '''
  mode: 0 for keeping some data frames from each movie in validation set, 1 for keeping complete movies in validation set
  split_ratio: fraction of data to be kept in validation/testing set. Should be between 0 and 1
  save: option to Save the training and validation splits to hdf5 file. Default value: False
  
  Output: two 4D numpy tensors/arrays containing positive samples, corresponding to training and validation splits respectively.
  '''
  pass

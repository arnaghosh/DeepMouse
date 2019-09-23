# Import stuff here
import numpy as np
from os import listdir
from os.path import isfile, join
import imageio
from scipy import ndimage
import matplotlib.pyplot as plt


def prepDataArrayFromMovie(movieName, num_cont_frames, overlap_delta = 0, temp_downsampling = 1, spatial_downsampling = 1):

  '''
  movieName: Name of the folder to extract data frames from
  num_cont_frames: Number of contiguous frames to append to create a data frame
  temp_downsampling: Temporal downsampling factor, picks every t_{th} frame from folder
  spatial_downsampling: Spatial downsampling factor, reduces image size by a factor of 's' (s < 1 for downsampling, s > 1 for upsampling)
  
  Output: a 4D numpy tensor/array containing data frames from specified movie, Size: [M_i, num_cont_frames, Height, Width]
          M_i is number of sequences from movie.
  '''
 
  
  catcamdataFolder = '/Users/shahab/Mila/Data/CatCam/'
  whichMovie = movieName
  
  folderpath = catcamdataFolder + whichMovie + '/'

  allframes = [f for f in listdir(folderpath) if isfile(join(folderpath, f))]
  numFrames = len(allframes)
  J = range(1,numFrames,int(1//temp_downsampling))
  availFrames = len(J)  
  numSamples = (availFrames - overlap_delta) // (num_cont_frames - overlap_delta)
  

  # loop over number of sequences extracted from the movie
  for i in range(numSamples):

    seqcounter = i + 1
    firstFrame = (seqcounter - 1) * num_cont_frames - (seqcounter - 1) * overlap_delta + 1
    lastFrame = (seqcounter) * num_cont_frames - (seqcounter - 1) * overlap_delta
    if lastFrame > len(allframes)-1:
        print(len(allframes))
        break
        
    # loop over single frames within a sequence
    
    for j in range(firstFrame, lastFrame + 1):
        
        whichImage = allframes[J[j]]
        imageloc = folderpath + whichImage
        thisframe = np.asarray(imageio.imread(imageloc))  
        thisframe = ndimage.zoom(thisframe, spatial_downsampling)
        
        
    
        
        if j == firstFrame:
            thisseq = np.array(thisframe,ndmin = 3)
        else:
            thisseq = np.append(thisseq,np.array(thisframe,ndmin = 3),axis=0)
    
    
    if i == 0:
        movie_seq = np.array(thisseq,ndmin = 4)
    else: 
        
        movie_seq = np.append(movie_seq,np.array(thisseq,ndmin = 4),axis = 0)
    
  return movie_seq
  
def  createTrainValidationSplits(mode, split_ratio, save=False):
  '''
  mode: 0 for keeping some data frames from each movie in validation set, 1 for keeping complete movies in validation set
  split_ratio: fraction of data to be kept in validation/testing set. Should be between 0 and 1
  save: option to Save the training and validation splits to hdf5 file. Default value: False
  
  Output: two 4D numpy tensors/arrays containing positive samples, corresponding to training and validation splits respectively.
  '''
  pass

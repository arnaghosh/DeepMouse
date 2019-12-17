from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pprint
import allensdk.brain_observatory.stimulus_info as stim_info
import numpy as np
from generate_SSM import *


boc = BrainObservatoryCache()
targeted_structures = boc.get_all_targeted_structures()
imaging_depths = boc.get_all_imaging_depths()

def get_RSM(CreLine = ['Cux2-CreERT2'], TargetedStruct = ['VISp'], ImagingDepth = [175], StimType = 'natural_scenes'):
    # get the number of exp containers (length of num_exps) and the number of experiments within each container (each element of num_exps)
    num_exps = get_number_experiments(CreLine = CreLine,TargetedStruct = TargetedStruct, ImagingDepth = ImagingDepth, StimType = StimType)
    num_exp_containers = len(num_exps)
    
    num_stim_conditions = get_num_stim_conditions(StimType = StimType)
    all_RSM = np.ndarray((num_stim_conditions,num_stim_conditions,sum(num_exps)))
    totalexp = 0
    for eccounter in range(0,num_exp_containers):
        for expcounter in range(0,num_exps[eccounter]):
            print(eccounter, expcounter)
            data_set = get_one_dataset(CreLine = CreLine, TargetedStruct = TargetedStruct, ImagingDepth = ImagingDepth, StimType = StimType, ExpContainerIdx = eccounter,ExpIdx = expcounter)
            if StimType == 'natural_scenes':
                activations = get_activations_natuscene(data_set)

            elif StimType == 'drifting_gratings':
                activations = get_activations_dg(data_set)

            elif StimType == 'static_gratings':
                activations = get_activations_sg(data_set)

            sim_mats = compute_similarity_matrices(activations)
            RSM_v1=sim_mats['V1']
            all_RSM[:,:,totalexp] = RSM_v1
            totalexp = totalexp + 1

    
    return all_RSM
            
    
def compare_two_RSMs(RSM1,RSM2):
    
    r=compute_ssm(RSM1, RSM2)
    return r

def compare_multi_RSMs(all_RSM):
    
    num_RSMs = all_RSM.shape[2]
    R = np.empty((num_RSMs,num_RSMs))
    for i in range(0,num_RSMs):
        for j in range(0,num_RSMs):
            R[i,j] = compare_two_RSMs(all_RSM[:,:,i],all_RSM[:,:,j])

            
    return R
    
    
def get_number_experiments(CreLine = ['Cux2-CreERT2'],TargetedStruct = ['VISp'],ImagingDepth = [175], StimType = 'natural_scenes'):
    all_ecs = boc.get_experiment_containers(cre_lines=CreLine,targeted_structures=TargetedStruct,imaging_depths=ImagingDepth)
    num_exp_containers = len(all_ecs)
    print("number of ", *CreLine, "experiment containers: %d\n" % num_exp_containers)
    num_exps = list()
    for eccounter in range(0,num_exp_containers):
        ec_id = all_ecs[eccounter]['id']
        exps = boc.get_ophys_experiments(experiment_container_ids=[ec_id], 
                                            stimuli=[StimType])
        
        num_exps.append(len(exps))
        print("experiment container: %d\n" % ec_id, ":", len(exps))
        
    return num_exps
    

def get_one_dataset(CreLine = ['Cux2-CreERT2'],TargetedStruct = ['VISp'],ImagingDepth = [175], StimType = 'natural_scenes', ExpContainerIdx = 0,ExpIdx = 0):
    
    all_ecs = boc.get_experiment_containers(cre_lines=CreLine,targeted_structures=TargetedStruct,imaging_depths=ImagingDepth)
    all_ec_id = all_ecs[ExpContainerIdx]['id']
    exp = boc.get_ophys_experiments(experiment_container_ids=[all_ec_id], 
                                        stimuli=[StimType])[ExpIdx]
        
    data_set = boc.get_ophys_experiment_data(exp['id'])
    
    return data_set

def get_num_stim_conditions(StimType = ['natural_scenes']):
    
    if StimType == 'static_gratings':
        all_orientations = [0,30,60,90,120,150]
        all_sf = [0.02,0.04,0.08,0.16,0.32]
        all_ph = [0,0.25,0.5,0.75]
        num_stim_conditions = len(all_orientations) * len(all_sf) * len(all_ph)
        
    elif StimType == 'drifting_gratings':
        all_directions = [0,45,90,135,180,225,270,315]
        all_tf = [1,2,4,8,15]
        num_stim_conditions = len(all_directions) * len(all_tf)
        
    elif StimType == 'natural_scenes':
        numImages = 118
        num_stim_conditions = numImages

    return num_stim_conditions

def get_activations_sg(data_set):
    stim_table = data_set.get_stimulus_table('static_gratings')
    all_cell_ids = data_set.get_cell_specimen_ids()
    num_neurons = len(all_cell_ids)
    print('there are ' + str(num_neurons) + ' neurons in this session')
    all_orientations = [0,30,60,90,120,150]
    all_sf = [0.02,0.04,0.08,0.16,0.32]
    all_ph = [0,0.25,0.5,0.75]

    responses = np.empty([num_neurons,len(all_orientations)*len(all_sf)*len(all_ph)])

    for ncounter in range(0,num_neurons):
        _, sample_cell = data_set.get_dff_traces(cell_specimen_ids=[all_cell_ids[ncounter]])
        sample_cell = sample_cell[0]
        counter = 0
        for sfcount, sf in enumerate(all_sf):
            for orcount, ori in enumerate(all_orientations):
                for pcount, ph in enumerate(all_ph):

                    thisstim = stim_table[(stim_table.spatial_frequency == sf) & (stim_table.orientation == ori) & (stim_table.phase == ph)].to_numpy()
                    response_tmp = np.empty([1,thisstim.shape[0]])
                    for tr in range(0,thisstim.shape[0]):
                        response_tmp[0,tr] =  np.nanmean(sample_cell[int(thisstim[tr,3]):int(thisstim[tr,4])])

                    responses[ncounter,counter] = np.median(response_tmp)
                    counter = counter + 1


    print(responses.shape)  

    activations = {'V1':np.transpose(responses)}
    
    return activations
    
    
def get_activations_dg(data_set):
    stim_table = data_set.get_stimulus_table('drifting_gratings')
    all_cell_ids = data_set.get_cell_specimen_ids()
    num_neurons = len(all_cell_ids)
    print('there are ' + str(num_neurons) + ' neurons in this session')
    all_directions = [0,45,90,135,180,225,270,315]
    all_tf = [1,2,4,8,15]

    responses = np.empty([num_neurons,len(all_directions)*len(all_tf)])

    for ncounter in range(0,num_neurons):
        _, sample_cell = data_set.get_dff_traces(cell_specimen_ids=[all_cell_ids[ncounter]])
        sample_cell = sample_cell[0]
        counter = 0
        for tfcount, tf in enumerate(all_tf):
            for dircount, direct in enumerate(all_directions):

                thisstim = stim_table[(stim_table.temporal_frequency == tf) & (stim_table.orientation == direct)].to_numpy()
                response_tmp = np.empty([1,thisstim.shape[0]])
                for tr in range(0,thisstim.shape[0]):
                    response_tmp[0,tr] =  np.nanmean(sample_cell[int(thisstim[tr,3]):int(thisstim[tr,4])])

                responses[ncounter,counter] = np.median(response_tmp)
                counter = counter + 1


    print(responses.shape)  

    activations = {'V1':np.transpose(responses)}
    
    return activations

def get_activations_natuscene(data_set):
    
    stim_table = data_set.get_stimulus_table('natural_scenes')
    all_cell_ids = data_set.get_cell_specimen_ids()
    num_neurons = len(all_cell_ids)
    print('there are ' + str(num_neurons) + ' neurons in this session')
    numImages = 118

    responses = np.empty([num_neurons,numImages])

    for ncounter in range(0,num_neurons):
        _, sample_cell = data_set.get_dff_traces(cell_specimen_ids=[all_cell_ids[ncounter]])
        sample_cell = sample_cell[0]
        counter = 0
        for imcounter in range(0,numImages):

            thisstim = stim_table[(stim_table.frame == imcounter)].to_numpy()
            response_tmp = np.empty([1,thisstim.shape[0]])

            for tr in range(0,thisstim.shape[0]):
                response_tmp[0,tr] =  np.nanmean(sample_cell[int(thisstim[tr,1]):int(thisstim[tr,2])])

            responses[ncounter,counter] = np.median(response_tmp)
            counter = counter + 1


    print(responses.shape)  

    activations = {'V1':np.transpose(responses)}
    
    return activations


if __name__=="__main__":

#     data = get_one_dataset(CreLine=['Cux2-CreERT2'],TargetedStruct=['VISp'],ImagingDepth=[175],ExpContainerIdx=0,ExpIdx=0)
#     num_exps = get_number_experiments(CreLine = ['Cux2-CreERT2'],TargetedStruct = ['VISp'],ImagingDepth = [175])
    all_RSM = get_RSM(CreLine = ['Cux2-CreERT2'], TargetedStruct = ['VISp'], ImagingDepth = [175], StimType = 'static_gratings')
    R = compare_multi_RSMs(all_RSM)
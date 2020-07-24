# -*- coding: utf-8 -*-
import logging
import os
import pickle
import random
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

# torch.autograd.set_detect_anomaly(True)# for detecting abnormal

def class_objtype(object_type, dataset='Apol'):# APOL
    if dataset == 'Apol':
        if object_type == 1 or object_type == 2:#Vehicle
            return 'Vehicle'
        elif object_type == 3:#Pedestrian
            return 'Pedestrian'
        elif object_type == 4:#Bicycle
            return 'Bicycle'
        else:
            return 'Unknown'

def GetDatasetIndInfo(dataset):
    #return f_ind, id_ind, x_ind, y_ind, yaw_ind, type_ind
    if dataset=='Apol':
        return 0, 1, 3, 4, 5, 2
    elif dataset=='Lyft':
        return 0, 1, 2, 3, 4, 5
    else:
        return 0, 1, 3, 4, 5, 2

def ExtractData(raw_data_dir, data_files, dataset):
    
    full_data_list = np.array([])
    track_data_list = {}
    min_position_x = 1000
    max_position_x = -1000
    min_position_y = 1000
    max_position_y = -1000
    
    gaussian_scaling = False
    x_gather = list()
    y_gather = list()
    # Generate datasetf
    print(data_files)
    f_ind, id_ind, x_ind, y_ind, yaw_ind, type_ind = GetDatasetIndInfo(dataset)

    for ind_directory, raw_file_name in enumerate(data_files):
        
        # for Lyft
        file_path = os.path.join(raw_data_dir, raw_file_name)# Each .txt or .csv file
        tmp0 = file_path.split('/')[-1].split('_')[1]
        tmp1 = file_path.split('/')[-1].split('_')[2]
        dataset_name = int(tmp0+tmp1.zfill(2))#only for Apolloscape dataset
        
        read = np.loadtxt(file_path, delimiter=' ')#APOL
        # read = np.load('file_path')#Lyft
        min_position_x = min(min_position_x, min(read[:, x_ind]))
        max_position_x = max(max_position_x, max(read[:, x_ind]))
        min_position_y = min(min_position_y, min(read[:, y_ind]))
        max_position_y = max(max_position_y, max(read[:, y_ind]))
        
        if gaussian_scaling:
            x_gather += read[:, x_ind].tolist()
            y_gather += read[:, y_ind].tolist()
        relevant_data = read[:, [f_ind, id_ind, type_ind, x_ind, y_ind, yaw_ind]]#(frame, id, type, x, y, heading)
        dataset_name_list = np.full([relevant_data.shape[0], 1], dataset_name) 
        # (dsId, frame, agnetId, type, x, y, heading)
        relevant_data = np.concatenate((dataset_name_list, relevant_data), axis=1)
        if full_data_list.size == 0:
            full_data_list = relevant_data
        else:
            full_data_list = np.concatenate((full_data_list, relevant_data), axis=0)

        agents = np.unique(read[:,id_ind])
        track_data = {}# Shape: agent x (frame, x, y)
        for agent in agents:
            a_track = read[read[:,id_ind]==agent][:,[id_ind, x_ind, y_ind]]
            track_data[agent] = a_track
        track_data_list[dataset_name] = track_data
    
    # Scaling!
    if gaussian_scaling:# Scale 'standard normal distribution'
        x_gather = np.array(x_gather)
        y_gather = np.array(y_gather)
        mean_x = np.mean(x_gather)
        mean_y = np.mean(y_gather)
        std_x = np.std(x_gather)
        std_y = np.std(y_gather)

        scale_param = (min_position_x, max_position_x, min_position_y, max_position_y)
        full_data_list[:, 4] = (full_data_list[:, 4] - mean_x) / std_x
        full_data_list[:, 5] = (full_data_list[:, 5] - mean_y) / std_y
        
        for dataset_name in track_data_list.keys():
            for agent in track_data_list[dataset_name].keys():
                track_data_list[dataset_name][agent][:, 1] = (track_data_list[dataset_name][agent][:, 1] - mean_x) / std_x                    
                track_data_list[dataset_name][agent][:, 2] = (track_data_list[dataset_name][agent][:, 2] - mean_y) / std_y
        scale_param = (mean_x, std_x, mean_y, std_y)
    
    else:# Scale range [-1, 1]
        full_data_list[:, 4] = (
                (full_data_list[:, 4] - min_position_x) / (max_position_x - min_position_x)
            ) * 2 - 1
        full_data_list[:, 5] = (
                (full_data_list[:, 5] - min_position_y) / (max_position_y - min_position_y)
            ) * 2 - 1
        
        for dataset_name in track_data_list.keys():
            for agent in track_data_list[dataset_name].keys():
                track_data_list[dataset_name][agent][:, 1] = (
                        (track_data_list[dataset_name][agent][:, 1] - min_position_x) / (max_position_x - min_position_x)
                    ) * 2 - 1
                track_data_list[dataset_name][agent][:, 2] = (
                        (track_data_list[dataset_name][agent][:, 2] - min_position_y) / (max_position_y - min_position_y)
                    ) * 2 - 1
        scale_param = (min_position_x, max_position_x, min_position_y, max_position_y)
        
    return (full_data_list, track_data_list, scale_param)

def GetScaleParam(raw_data_dir, data_files, dataset_name):
    min_position_x = 10000
    max_position_x = -10000
    min_position_y = 10000
    max_position_y = -10000
    
    # Generate datasetf
    f_ind, id_ind, x_ind, y_ind, yaw_ind, type_ind = GetDatasetIndInfo(dataset_name)

    for ind_directory, raw_file_name in enumerate(data_files):
        file_path = os.path.join(raw_data_dir, raw_file_name)# Each .txt or .csv file
        read = np.load(file_path) if dataset_name=='Lyft' else np.genfromtxt(file_path)#Apol

        min_position_x = min(min_position_x, min(read[:, x_ind]))
        max_position_x = max(max_position_x, max(read[:, x_ind]))
        min_position_y = min(min_position_y, min(read[:, y_ind]))
        max_position_y = max(max_position_y, max(read[:, y_ind]))
    
    return (min_position_x, max_position_x, min_position_y, max_position_y)

def PreprocessData(args):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    tr_ratio, val_ratio, te_ratio = args.train_val_test_ratio
    train_fraction = tr_ratio + val_ratio
    val_fraction = val_ratio
    print("Data fraction is (train: {}%, val: {}%, test: {}%)"\
            .format(tr_ratio*100, val_ratio*100, te_ratio*100))

    # List of data directories where raw data resides
    raw_data_dir = args.raw_data_dir
    dataset_cnt = len(os.listdir(raw_data_dir))
    dataset_idx = sorted(os.listdir(raw_data_dir))
    np.random.shuffle(dataset_idx)# By random seed.

    # Divide the datasets to {train, val, test}
    data_dir_train = dataset_idx[: int(dataset_cnt * tr_ratio)]
    data_dir_val = dataset_idx[int(dataset_cnt * tr_ratio): int(dataset_cnt * train_fraction)]
    data_dir_test = dataset_idx[int(dataset_cnt * train_fraction) :]
    train_scale_param = GetScaleParam(raw_data_dir, data_dir_train, args.dataset_name)
    val_scale_param = GetScaleParam(raw_data_dir, data_dir_val, args.dataset_name)
    test_scale_param = GetScaleParam(raw_data_dir, data_dir_test, args.dataset_name)

    # Save dataset path corresponding that seed as the pickle file
    f = open(args.data_file, "wb")
    pickle.dump(
        ((data_dir_train,train_scale_param), (data_dir_val, val_scale_param), (data_dir_test, test_scale_param)),
        f,
        protocol=2,
    )
    f.close()
    
# Dataset 상속
''' 
For Lyft dataset::
Data config. of each row --> [frame_idx, agent_id, x, y, yaw, agent_type]
We divde the whole dataset into 20 minute each. (Total driving time is 112 hours at train_0 dataset)
Using agent_type:
| 1: PEDESTRIAN
| 2: BICYCLE, MOTORCYCLE, CYCLIST, MOTORCYCLIST
| 3: CAR, VAN, TRAM, OTHER_VEHICLE
| 4: BUS, TRUCK, EMERGENCY_VEHICLE
| 5: UNKNOWN
'''
class DataSet(Dataset):   
    def __init__(self, args, dtype):

        # Store the arguments
        self.batch_size = args.batch_size
        self.obs_length = args.obs_length
        self.pred_length = args.pred_length
        self.use_cuda = args.cuda
        self.device = args.device

        self.dtype = dtype
        self.raw_data_dir = args.raw_data_dir
        self.total_data_num = 0# the total number of data
        self.dataset_len_list = None# cumulative data length for all data
        self.args = args
        
        self.min_position_x, self.max_position_x, self.min_position_y, self.max_position_y = None, None, None, None
        self.load_pickle(args.data_file)
        os_path = os.path.join("../data/", "{}-dataset-seed{}-{}_index.npy".format(self.args.dataset_name, self.args.seed, self.dtype))#pickle contains 'train', 'val', 'test' file name and 'scale param'
        if not os.path.exists(os_path):
            self.generate_dataset_ind()
        self.dataset_len_list = np.load(os_path).astype(int)
        self.total_data_num = self.dataset_len_list[-1]
        
        self.f_ind, self.id_ind, self.x_ind, self.y_ind, self.yaw_ind, self.type_ind = GetDatasetIndInfo(args.dataset_name)
        
        self.current_dataset = None
        self.current_data = None
        self.f_interval  = args.frame_interval


    def load_pickle(self, data_dir):
        f = open(data_dir, "rb")
        self.raw_data_path = pickle.load(f)
        f.close()
        # Get 'dtype' data from the pickle file
        if self.dtype == 'train':
            self.full_data_path, self.scale_param = self.raw_data_path[0]
        elif self.dtype == 'val':
            self.full_data_path, self.scale_param = self.raw_data_path[1]
        else:# test
            self.full_data_path, self.scale_param = self.raw_data_path[2]
        self.min_position_x, self.max_position_x, self.min_position_y, self.max_position_y = self.scale_param
        print("Load the dataset... (total#: {})".format(len(self.full_data_path)))

    def generate_dataset_ind(self):
        self.dataset_len_list = np.zeros(len(self.full_data_path))
        for ind, raw_file_name in enumerate(self.full_data_path):
            file_path = os.path.join(self.raw_data_dir, raw_file_name)# Each .txt or .csv file
            read = np.load(file_path) if self.args.dataset_name == 'Lyft' else np.genfromtxt(file_path)#Lyft
            self.total_data_num += len(read)
            self.dataset_len_list[ind] = self.total_data_num

        with open("../data/{}-dataset-seed{}-{}_index.npy".format(self.args.dataset_name, self.args.seed, self.dtype), 'wb') as f:
            np.save(f, self.dataset_len_list)
        print("Successfully, generate ind Info. and save the data")
        print("# of data: ", self.total_data_num)
        

    def GetDatasetFile(self, idx):
        prev_dataset_len = 0
        dataset_ind = 0
        data_ind = 0
        for ind, dataset_len in enumerate(self.dataset_len_list):
            if idx >= dataset_len:
                prev_dataset_len = dataset_len
                continue
            else:
                dataset_ind = ind
                data_ind = idx - prev_dataset_len
                break
        dataset_path = os.path.join(self.raw_data_dir, self.full_data_path[dataset_ind])
        
        # get current dataset
        self.current_dataset = np.load(dataset_path) if self.args.dataset_name == 'Lyft' else np.genfromtxt(dataset_path)#Apol
        self.normalize_data()# Normalized a position <x,y> in a range of [-1, +1]

        self.current_data = self.current_dataset[data_ind]

        return dataset_path, data_ind

    # 총 데이터의 개수를 리턴 (여기서는 each frame of each agent가 하나의 data point)
    def __len__(self):
        return self.total_data_num

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        dataset_path, data_ind = self.GetDatasetFile(idx)
        
        dsId = dataset_path# dataset Id
        frame = self.current_data[self.f_ind].astype(int)# frame in the dataset Id
        agentId = self.current_data[self.id_ind].astype(int)# unique agentID in the dataset
        agentType = self.current_data[self.type_ind].astype(int)
        # pose = self.current_data[[self.x_ind, self.y_ind, self.yaw_ind]]# [x, y, yaw]
        AgentInfo = (dsId, frame, agentId, agentType)
        # if agentType == 3:
        # print(AgentInfo)
        hist, ref_pose, hist_mask = self.get_history(agentId, frame, agentId, dsId)
            # print(hist)
        fut = self.get_future(agentId, frame, dsId)
        fut_mask = len(fut)
        # print("DEL!")
        del self.current_dataset
    
        return hist, hist_mask, fut, fut_mask, ref_pose, AgentInfo

    def get_history(self, agentId, frame, ref_agentId, dsId):

        # Based on the reference trajectory, get a relative trajectory
        ref_track = self.current_dataset[self.current_dataset[:,self.id_ind]==agentId].astype(float)
        ref_pose = ref_track[ref_track[:,0]==frame][0, [self.x_ind, self.y_ind]]
        
        agent_track = self.current_dataset[self.current_dataset[:,self.id_ind]==agentId].astype(float)
        
        # for setting interval
        obs_length = int(np.argwhere(agent_track[:, 0] == frame).item(0)/self.f_interval + 1) if np.argwhere(agent_track[:, 0] == frame).item(0) - (self.obs_length-1)*(self.f_interval) < 0 else self.obs_length
        start_idx = np.maximum(0, np.argwhere(agent_track[:, 0] == frame).item(0) - (obs_length-1)*self.f_interval)
        end_idx = np.argwhere(agent_track[:, 0] == frame).item(0) + 1
        # print("hist::::")
        # print("frame: ", agent_track[start_idx:end_idx:self.f_interval, 0])
        hist = agent_track[start_idx:end_idx:self.f_interval,[self.x_ind, self.y_ind]] - ref_pose# Get only relative positions [m]
        reasonable_inds = np.where(agent_track[start_idx:end_idx:self.f_interval, 0]>=frame-self.f_interval*(self.obs_length-1))[0]
        # print(reasonable_inds)
        hist = hist[reasonable_inds]
        # print("HIST: ", hist)
        hist_mask = len(hist)
        if len(hist) < self.obs_length:
            tmp0 = np.full((self.obs_length,2), hist[0])
            # tmp0 = np.full((self.obs_length,2), 1e-6)
            tmp0[tmp0.shape[0]-hist.shape[0]:,:] = hist
            return tmp0, ref_pose, hist_mask

        return hist, ref_pose, hist_mask

    def get_future(self, agentId, frame, dsId):
        agent_track = self.current_dataset[self.current_dataset[:,self.id_ind]==agentId].astype(float)
        ref_pose = agent_track[agent_track[:,0]==frame][0, [self.x_ind, self.y_ind]]

        start_idx = np.argwhere(agent_track[:, 0] == frame).item(0)+self.f_interval#t+1 frame
        end_idx = np.minimum(len(agent_track), np.argwhere(agent_track[:, 0] == frame).item(0) + self.pred_length*self.f_interval + 1)#t+future frame
        # start_idx = np.argwhere(agent_track[:, 0] == frame).item(0)+1#t+1 frame
        # end_idx = np.minimum(len(agent_track), np.argwhere(agent_track[:, 0] == frame).item(0) + self.pred_length + 1)#t+future frame
        # print("fut::::")
        # print("frame: ", agent_track[start_idx:end_idx:self.f_interval, 0])
        fut = agent_track[start_idx:end_idx:self.f_interval,[self.x_ind, self.y_ind]] - ref_pose
        reasonable_inds = np.where(agent_track[start_idx:end_idx:self.f_interval, 0]<=frame+self.f_interval*self.pred_length)[0]
        fut = fut[reasonable_inds]
        # print("FUT: ", fut)
        return fut


    def GetBatch(self, samples):
        
        # Filtering bad data sample// It can be modified, for filtering data which you don't want to involve
        # samples = [sample_pt for sample_pt in samples if sample_pt[-1][-1] == 3]
        # while self.batch_size > len(samples):
        #     new_sample = self[np.random.randint(0, len(self))]
        #     if new_sample[-1][-1] == 3:
        #         samples.append(new_sample)
        # print(len(samples))
        
        # quit()
        # Initialization
        hist_batch = torch.zeros(len(samples), self.obs_length, 2)
        fut_batch = torch.zeros(len(samples), self.pred_length, 2)
        fut_mask_batch = torch.zeros(len(samples), self.pred_length, 2)
        ref_pose_batch = torch.zeros(len(samples), 2)
        AgentsInfo = [None]*len(samples)
        for sampleId, (hist, hist_mask, fut, fut_mask, ref_pose, AgentInfo) in enumerate(samples):
            hist_batch[sampleId, :, 0] = torch.from_numpy(hist[:, 0])# x
            hist_batch[sampleId, :, 1] = torch.from_numpy(hist[:, 1])# y
            fut_batch[sampleId, 0:len(fut), 0] = torch.from_numpy(fut[:, 0])# x
            fut_batch[sampleId, 0:len(fut), 1] = torch.from_numpy(fut[:, 1])# y
            fut_mask_batch[sampleId, 0:len(fut), :] = 1# future (x,y) exist or not?
            ref_pose_batch[sampleId, 0] = ref_pose[0]
            ref_pose_batch[sampleId, 1] = ref_pose[1]
            AgentsInfo[sampleId] = AgentInfo
            
        return hist_batch, fut_batch, fut_mask_batch, ref_pose_batch, AgentsInfo

    def normalize_data(self):
        self.current_dataset[:, self.x_ind] = (
                (self.current_dataset[:, self.x_ind] - self.min_position_x) / (self.max_position_x - self.min_position_x)
            ) * 2 - 1
        self.current_dataset[:, self.y_ind] = (
                (self.current_dataset[:, self.y_ind] - self.min_position_y) / (self.max_position_y - self.min_position_y)
            ) * 2 - 1

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask, device='cpu'):
    acc = torch.zeros_like(mask, device=device)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    
    # fde_weight = 1
    # tmp0 = mask.cpu().numpy()
    # for i in range(len(tmp0)):
    #     if list(np.where(tmp0[i][:,0]==1))[0] != []:
    #         acc[i,list(np.where(tmp0[i][:,0]==1))[0][-1]]*=fde_weight
    #     else:
    #         continue
    # acc[i,list(np.where(tmp0[i][:,0]==1))[0][-1]]*=fde_weight
    # print
    lossVal = torch.sum(acc)/torch.sum(mask)
    
    return lossVal


def maskedLastPositionLoss(y_pred, y_gt, mask, device='cpu'):
    
    masking_count = 0
    lossVal = 0
    batch_size = y_pred.shape[0]
    mask = torch.zeros((mask.shape)).cuda()
    for batch_ind in range(batch_size):
        masked_pred = y_pred[batch_ind][y_pred[batch_ind][:,0]!=0]
        if masked_pred.shape[0] == 0:#no GT exists
            continue
        # Sampling
        mean = [masked_pred[-1][0].item(), masked_pred[-1][1].item()]
        cov = [[masked_pred[-1][2].item()**2, 0],\
               [0, masked_pred[-1][3].item()**2]]
        
        sample_x, sample_y = np.random.multivariate_normal(mean, cov, 1).T
        y_pred[batch_ind][masked_pred.shape[0]-1] = torch.tensor([sample_x[0], sample_y[0], 0., 0.]).cuda()
        mask[batch_ind][masked_pred.shape[0]-1] = torch.tensor([1,1,1,1]).cuda()#only considers last position
    
    acc = torch.zeros_like(mask, device=device)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc = acc[:,:,:1]
    mask = mask[:,:,:1]
    acc = acc*mask

    lossVal = torch.sum(acc)/torch.sum(mask)
    
    return lossVal
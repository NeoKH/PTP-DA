#!/usr/bin/env python3
# coding:utf8
"""
@file      eth_ucy_base.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

'''
base dataset process, just split and convert coordinate, not convert to graph
'''
import os,sys
from re import A
import random
import math
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import numpy as np
from torch.utils.data import Dataset

from utils.data import poly_fit,angles_cluster

class ETH_UCY_BaseData(Dataset):
    """ETH_UCY_BaseData
    
    """
    def __init__(
        self,
        target_name:str,
        data_path:str,
        len_observe:int=8,
        len_predict:int=12,
        skip:int=1,
        min_ped:int=2,
        threshold:float=0.002,
        delim:str='\t',
        relative_fun:str="self_offset",
        map_path:str=None,
        augment:List=None, 
        rot_type:str="cluster"
    ):
        """
        Args:
        - data_path: Directory containing dataset files in the format <frame_id> <ped_id> <x> <y>
        - len_observe: Number of time-steps in input trajectories
        - len_predict: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - threshold: Minimum error to be considered for non linear traj when using a linear predictor
        - delim: Delimiter in the dataset files
        - relative_fun: transfer coordinates into relative coordinates, "self_offset" or "last_frame_mean"
        - map_path: load static scence map or not
        - augment: whether or not to augment data.
        """
        super(ETH_UCY_BaseData, self).__init__()
        self.target_name = target_name
        self.data_path = data_path
        
        self.len_observe = len_observe
        self.len_predict = len_predict
        self.len_total = self.len_observe + self.len_predict
        
        self.skip = skip
        self.delim = delim
        self.augment = augment if not augment is None else []
        self.relative_fun = relative_fun
        self.map_path = map_path
        self.min_ped = min_ped

        self.max_peds_in_scene = 0

        all_files = os.listdir(self.data_path)
        all_files = [os.path.join(self.data_path, _path) for _path in all_files]

        num_peds_in_seq = []
        mean_centers_in_seq = []
        seq_list = []
        seq_rel_list = []
        temporal_mask_list = []
        non_linear_mask_list = []
        
        if self.augment:
            thetas = angles_cluster(name=self.target_name,)
        
        for path in all_files:
            data = self.read_eth_ucy(path,delim) # <frame_id> <ped_id> <x> <y>
            frames = np.unique(data[:,0]).tolist() # all frame timestamps

            # split data into sequence
            data_frames = []
            for frame in frames:
                data_frames.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.len_total + 1) / self.skip))
            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                # deals with single sequence
                data_sequence = np.concatenate(data_frames[idx:idx + self.len_total], axis=0)

                peds_in_seq = np.unique(data_sequence[:, 1])
                self.max_peds_in_scene = max(self.max_peds_in_scene,len(peds_in_seq))

                if "rotate" in self.augment:
                    # define rotate matrixs
                    rotate_matrixs = []
                    rotate_matrixs.append(np.eye(2,dtype=float))
                    if rot_type=="random":
                        for ii in range(0,4):
                            theta = random.uniform(ii/2*np.pi,(ii+1)/2*np.pi)
                            rot = np.asarray([
                                [np.cos(theta),-np.sin(theta)],
                                [np.sin(theta),np.cos(theta)]
                            ],dtype=float)
                            rotate_matrixs.append(rot)
                    else:
                        for theta in thetas:
                            rot = np.asarray([
                                [np.cos(theta),-np.sin(theta)],
                                [np.sin(theta),np.cos(theta)]
                            ],dtype=float)
                            rotate_matrixs.append(rot)
                    rotate_matrixs = np.asarray(rotate_matrixs,dtype=float)
                    augment_len = rotate_matrixs.shape[0]
                    _seqs        = np.zeros((augment_len,len(peds_in_seq), 2, self.len_total)) # (5,Np,2,L)
                    _seq_rels    = np.zeros((augment_len,len(peds_in_seq), 2, self.len_total)) # (5,Np,2,L)
                    _temporal_masks   = np.zeros((augment_len,len(peds_in_seq),self.len_total)) # (5,Np,L)
                    _non_linear_masks = np.zeros((augment_len,len(peds_in_seq))) # (5,Np,) 1 -> Non Linear 0-> Linear
                else:
                    _seq        = np.zeros((len(peds_in_seq), 2, self.len_total)) # (Np,2,L)
                    _seq_rel    = np.zeros((len(peds_in_seq), 2, self.len_total)) # (Np,2,L)
                    _temporal_mask   = np.zeros((len(peds_in_seq),self.len_total)) # (Np,L)
                    # _non_linear_mask = [] # (Np,) 1 -> Non Linear 0-> Linear
                    _non_linear_mask = np.zeros(len(peds_in_seq)) # (Np,) 1 -> Non Linear 0-> Linear
                    
                num_peds = 0
                for _pid,ped_id in enumerate(peds_in_seq):
                    # deals with single pedestrian
                    ped_seq = data_sequence[data_sequence[:, 1] ==ped_id, :]
                    ped_seq = np.around(ped_seq, decimals=4)

                    ped_seq_start = frames.index(ped_seq[0, 0]) - idx
                    ped_seq_end = frames.index(ped_seq[-1, 0]) - idx + 1
                    
                    if ped_seq_end - ped_seq_start < self.len_total:
                        # TODO
                        continue
                        if ped_seq_start >= self.len_observe - 1:
                            _observe_mask[_pid] = 0
                            _predict_mask[_pid] = 0
                        if ped_seq_end <= self.len_observe + 1:
                            _observe_mask[_pid] = 1
                            _predict_mask[_pid] = 0

                    ped_seq = np.transpose(ped_seq[:, 2:]) # (20,2) => (2,20)
                    if "rotate" in self.augment:
                        ped_seqs = np.matmul(rotate_matrixs,ped_seq)
                        # Make coordinates relative
                        if self.relative_fun == "self_offset":
                            rel_curr_ped_seqs = np.zeros_like(ped_seqs)
                            rel_curr_ped_seqs[:, :, 1:] = ped_seqs[:, :, 1:] - ped_seqs[:,:, :-1]
                            _seq_rels    [:,num_peds, :, ped_seq_start:ped_seq_end] = rel_curr_ped_seqs
                        _seqs[:, num_peds, :, ped_seq_start:ped_seq_end] = ped_seqs
                        _temporal_masks[:,num_peds, ped_seq_start:ped_seq_end]= 1
                        # Linear vs Non-Linear Trajectory
                        # Return: 1 -> Non Linear 0-> Linear
                        _non_linear_masks[:,num_peds] = poly_fit(ped_seq, len_predict, threshold)
                        pass
                    else:
                        # Make coordinates relative
                        if self.relative_fun == "self_offset":
                            rel_curr_ped_seq = np.zeros_like(ped_seq)
                            rel_curr_ped_seq[:, 1:] = ped_seq[:, 1:] - ped_seq[:, :-1]
                            _seq_rel    [num_peds, :, ped_seq_start:ped_seq_end] = rel_curr_ped_seq

                        _seq[num_peds, :, ped_seq_start:ped_seq_end] = ped_seq
                        _temporal_mask[num_peds, ped_seq_start:ped_seq_end]= 1

                        _non_linear_mask[num_peds] = poly_fit(ped_seq, len_predict, threshold)

                    num_peds +=1
                # TODO
                if num_peds < self.min_ped:
                    continue

                if self.relative_fun == "last_frame_mean":
                    # Make relative coordinates
                    mean_center = np.mean(_seq[:num_peds,:,self.len_observe-1], axis=0)
                    _seq_rel -= mean_center[None,:,None]
                    mean_centers_in_seq.append(mean_center[None,:])
                
                if "rotate" in self.augment:
                    for k in range(_seqs.shape[0]):
                        num_peds_in_seq.append(num_peds)
                        seq_list.append(_seqs[k,:num_peds])
                        seq_rel_list.append(_seq_rels[k,:num_peds])
                        temporal_mask_list.append(_temporal_masks[k,:num_peds])
                        non_linear_mask_list.append(_non_linear_masks[k,:num_peds])
                    pass
                else:
                    num_peds_in_seq.append(num_peds)
                    seq_list.append(_seq[:num_peds])
                    seq_rel_list.append(_seq_rel[:num_peds])
                    temporal_mask_list.append(_temporal_mask[:num_peds])
                    non_linear_mask_list.append(_non_linear_mask[:num_peds])
        
        self.num_seq = len(seq_list)
        
        seq_list            = np.concatenate(seq_list, axis=0) # [(N1,2,L),(N2,2,L),...] ==> (N1+N2+..., 2,L)
        seq_rel_list        = np.concatenate(seq_rel_list, axis=0) # [(N1,2,L),(N2,2,L),...] ==> (N1+N2+..., 2,L)
        temporal_mask_list  = np.concatenate(temporal_mask_list, axis=0) # [(N1,L),(N2,L),...] ==> (N1+N2+..., L)
        non_linear_mask_list= np.concatenate(non_linear_mask_list,axis=0) #
        
        self.observes = torch.from_numpy(seq_list[:,:,:self.len_observe]).type(torch.float)
        self.predicts = torch.from_numpy(seq_list[:,:,self.len_observe:]).type(torch.float)
        
        self.observes_rel = torch.from_numpy(seq_rel_list[:,:,:self.len_observe]).type(torch.float)
        self.predicts_rel = torch.from_numpy(seq_rel_list[:,:,self.len_observe:]).type(torch.float)
        
        self.temporal_mask = torch.from_numpy(temporal_mask_list).type(torch.float)
        self.non_linear_mask = torch.from_numpy(non_linear_mask_list).type(torch.float)
        
        if self.relative_fun=="last_frame_mean":
            mean_centers_in_seq = np.concatenate(mean_centers_in_seq,axis=0) #  ==> (N1+N2+..., 2)
            self.mean_centers = torch.from_numpy(mean_centers_in_seq).type(torch.float)
        
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def read_eth_ucy(self,_path, delim='\t'):
        data = []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = {
            "observe": self.observes[start:end, :],
            "predict": self.predicts[start:end, :],
            "observe_rel": self.observes_rel[start:end, :],
            "predict_rel": self.predicts_rel[start:end, :],
            "temporal_mask": self.temporal_mask[start:end, :],
            "non_linear_mask": self.non_linear_mask[start:end],
        }

        if self.relative_fun=="last_frame_mean":
            out["mean_center"] = self.mean_centers[index]

        return out

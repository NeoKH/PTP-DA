#!/usr/bin/env python3
# coding:utf8
"""
@file      unified_data.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import os
import math
from typing import Any, Callable, List, Optional, Tuple, Union

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
# from utils.data import poly_fit
from utils.data import read_eth_ucy,read_stcrowd

class UnifiedData(Dataset):
    """
    Unified dataset for ETH/UCY and STCrowd and so on
    """
    def __init__(
        self,
        data_root:str ="./data/origin",
        archive_name: str="ETH_UCY_DA",
        data_name : str = "eth",
        data_type: str = "test",
        
        len_observe:int=8,
        len_predict:int=12,
        skip:int=1,
        min_ped:int=2,
        threshold:float=0.002,
        relative_type:str="self_offset",
    ):
        """
        Args:
        - data_name: dataset name
        - data_type: 'train' 'val' or 'test'
        - src_root: File ; Directory containing dataset files
        - len_observe: Number of time-steps in input trajectories
        - len_predict: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - threshold: Minimum error to be considered for non linear traj when using a linear predictor
        - relative_type: transfer coordinates into relative coordinates, "self_offset" or "last_frame_mean"
        """
        super(UnifiedData, self).__init__()
        # Path check
        self.archive_name = archive_name
        self.data_name = data_name
        self.data_type = data_type
        self.data_path = Path(data_root) / archive_name / data_name / data_type
        assert self.data_path.exists()
        
        self.len_observe = len_observe
        self.len_predict = len_predict
        self.len_total = self.len_observe + self.len_predict
        
        self.skip = skip
        self.relative_type = relative_type

        self.min_ped = min_ped
        self.threshold = threshold
        self.max_peds_in_scene = 0
        
        num_peds_in_seq = []
        mean_centers_in_seq = []
        seq_list = []
        seq_rel_list = []
        temporal_mask_list = []
        non_linear_mask_list = []
        
        all_files = os.listdir(self.data_path)
        all_files = [os.path.join(self.data_path, _path) for _path in all_files]
        for p_file in all_files:
            if self.archive_name.lower().find("eth")!=-1:
                data = read_eth_ucy(p_file)
            else:
                data = read_stcrowd(p_file)
                
            # <frame_id> <ped_id> <x> <y>
            frames = np.unique(data[:,0]).tolist() # all frame timestamps
            
            # split data into sequence
            data_frames = []
            for frame in frames:
                data_frames.append(data[frame == data[:, 0], :])
            
            num_sequences = int(math.ceil((len(frames) - self.len_total + 1) / self.skip))
            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                '''处理单个Sequence'''
                data_sequence = np.concatenate(data_frames[idx:idx + self.len_total], axis=0)

                peds_in_seq = np.unique(data_sequence[:, 1])
                self.max_peds_in_scene = max(self.max_peds_in_scene,len(peds_in_seq))

                _seq        = np.zeros((len(peds_in_seq), 2, self.len_total)) # (Np,2,L)
                _seq_rel    = np.zeros((len(peds_in_seq), 2, self.len_total)) # (Np,2,L)
                
                num_peds = 0
                for _pid,ped_id in enumerate(peds_in_seq):
                    '''处理单个行人'''
                    ped_seq = data_sequence[data_sequence[:, 1] ==ped_id, :]
                    ped_seq = ped_seq.astype(np.float64)
                    ped_seq = np.around(ped_seq, decimals=4)

                    ped_seq_start = frames.index(ped_seq[0, 0]) - idx
                    ped_seq_end = frames.index(ped_seq[-1, 0]) - idx + 1
                    
                    if ped_seq_end - ped_seq_start < self.len_total:
                        # TODO
                        continue
                        # 当seq小于(past+futuer)的长度时
                        if ped_seq_start >= self.len_observe - 1:
                            _observe_mask[_pid] = 0
                            _predict_mask[_pid] = 0
                        if ped_seq_end <= self.len_observe + 1:
                            _observe_mask[_pid] = 1
                            _predict_mask[_pid] = 0

                    ped_seq = np.transpose(ped_seq[:, 2:]) # (20,2) => (2,20)
                    # Make coordinates relative
                    if self.relative_type == "self_offset":
                        rel_curr_ped_seq = np.zeros_like(ped_seq)
                        rel_curr_ped_seq[:, 1:] = ped_seq[:, 1:] - ped_seq[:, :-1]
                        _seq_rel    [num_peds, :, ped_seq_start:ped_seq_end] = rel_curr_ped_seq

                    _seq[num_peds, :, ped_seq_start:ped_seq_end] = ped_seq

                    # Linear vs Non-Linear Trajectory
                    # 1 -> Non Linear 0-> Linear

                    num_peds +=1

                # TODO
                if num_peds < self.min_ped:
                    continue
                num_peds_in_seq.append(num_peds)
                seq_list.append(_seq[:num_peds])
                seq_rel_list.append(_seq_rel[:num_peds])
        
        self.num_seq = len(seq_list)
        
        self.seq       = np.concatenate(seq_list, axis=0) # [(N1,2,L),(N2,2,L),...] ==> (N1+N2+..., 2,L)
        self.seq_rel   = np.concatenate(seq_rel_list, axis=0) # [(N1,2,L),(N2,2,L),...] ==> (N1+N2+..., 2,L)
        
        if self.relative_type=="last_frame_mean":
            mean_centers_in_seq = np.concatenate(mean_centers_in_seq,axis=0) #  ==> (N1+N2+..., 2)
            self.mean_centers = mean_centers_in_seq
        
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = {
            "observe": torch.tensor(self.seq[start:end,:, :self.len_observe]),
            "predict": torch.tensor(self.seq[start:end,:, self.len_observe:]),
        }
        return out



#!/usr/bin/env python3
# coding:utf8
"""
@file      unified_graph.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import math
import random

# import pickle
import torch
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path

from datasets.unified_data import UnifiedData
from utils.data import angle_statistic,scale_statistic,get_hist
from utils.data import get_sequence_average_angle,get_sequence_angle

class UnifiedGraph(UnifiedData):
    def __init__(
        self, 
        
        data_root: str = "./data/origin",
        archive_name: str="ETH_UCY_DA",
        data_name: str = "eth", 
        data_type: str = "test", 
        
        len_observe: int = 8, 
        len_predict: int = 12, 
        
        skip: int = 1,
        min_ped: int = 2, 
        threshold: float = 0.002, 
        adj_type = "anorm",
        relative_type: str = "self_offset",
        norm_lap_matr = True,
        
        augment:str= "none",
        target_name:str='eth',
        target_type :str="val",
        
        # angle augment params
        angle_bins:int = 45,
        angle_prob:float=0.7,
        angle_obs_len:int=8,
        angle_regenerate: bool = False,
        angle_csv:str='./data/statistic/angle'
    ):
        super(UnifiedGraph,self).__init__(
            data_root,archive_name,data_name, data_type,
            len_observe, len_predict, 
            skip, min_ped, threshold, relative_type
        )
        
        self.adj_type = adj_type
        self.norm_lap_matr = norm_lap_matr
        
        # augmentation config
        self.augment = augment.lower()
        self.target_name = target_name
        self.angle_bins = angle_bins # only work in target angle augment
        self.angle_prob = angle_prob # only work in target angle augment
        self.angle_obs_len = angle_obs_len # only work in target angle augment
        self.angle_csv = Path(angle_csv) / archive_name / target_type / f"{target_name}.pkl" # only work in target angle augment
        
        if not self.angle_csv.exists() or angle_regenerate:
            angle_statistic(
                src_root=data_root,
                dst_root=angle_csv,
                archive_name=archive_name,
                data_name=target_name,
                data_type = target_type
            )
        if self.augment=="angle_target":
            self.angle_hist = get_hist(self.angle_csv,self.angle_bins,self.angle_obs_len)
            self.angle_hist["prob"] = torch.tensor(self.angle_hist["prob"])

        self.random_angles = list(range(-180,180,30))
        
    def __getitem__(self, index, weight=1.0):
        self.weight = weight
        start, end = self.seq_start_end[index]
        seq = self.seq[start:end,:]
        seq_rel = self.seq_rel[start:end,:] # (N,2,T)

        if self.augment.find("angle")!=-1:
            # Angle Augmentation
            coin = random.random() # 以一定概率选择是否增广
            if  coin < self.angle_prob:
                if self.augment.find("target")!=-1:
                    # get average angle over the observe sequence
                    self.source_angle = get_sequence_average_angle(seq_rel[:,:,1:self.len_observe])
                    self.target_angle = self.get_target_angle()
                    self.rotate_angle = self.target_angle - self.source_angle
                    rotate_mat = np.asarray([
                        [np.cos(self.rotate_angle),-np.sin(self.rotate_angle)],
                        [np.sin(self.rotate_angle),np.cos(self.rotate_angle)]
                    ],dtype=float)
                    seq = np.matmul(rotate_mat,seq)
                    seq_rel = np.matmul(rotate_mat,seq_rel)
                if self.augment.find("random")!=-1:
                    self.rotate_angle = random.sample(self.random_angles,1)[0]
                    rotate_mat = np.asarray([
                        [np.cos(self.rotate_angle),-np.sin(self.rotate_angle)],
                        [np.sin(self.rotate_angle),np.cos(self.rotate_angle)]
                    ],dtype=float)
                    seq = np.matmul(rotate_mat,seq)
                    seq_rel = np.matmul(rotate_mat,seq_rel)

        angles = get_sequence_angle(seq_rel)
        
        observes = seq[:,:,:self.len_observe]
        predicts = seq[:,:,self.len_observe:]
        
        observes_rel = seq_rel[:,:,:self.len_observe]
        predicts_rel = seq_rel[:,:,self.len_observe:]
        
        V_obs,A_obs = self.seq_to_graph(observes_rel, need_A=True)
        V_pred, _   = self.seq_to_graph(predicts_rel, need_A=False)
        
        out = {
            "observe": torch.tensor(observes).type(torch.float),
            "predict": torch.tensor(predicts).type(torch.float),
            "observe_rel" : torch.tensor(observes_rel).type(torch.float),
            "predicts_rel" : torch.tensor(predicts_rel).type(torch.float),
            "V_observe" : torch.tensor(V_obs).type(torch.float),
            "A_observe" : torch.tensor(A_obs).type(torch.float),
            "V_predict" : torch.tensor(V_pred).type(torch.float),
            "angle": angles,
        }
        return out
    
    def seq_to_graph(self,seq,need_A=True):
        if self.adj_type == "anorm":
            seq_len = seq.shape[2]
            max_nodes = seq.shape[0]
            
            V = seq.transpose(2,0,1)
            if need_A:
                I = np.eye(max_nodes)
                Z = np.zeros((seq_len,max_nodes,max_nodes))
                dist =  np.expand_dims(V,2) - np.expand_dims(V,1)
                A = np.sqrt((dist**2).sum(3))
                A = A + I
                A = 1/A
                A = np.where(A==np.inf,Z,A)
                A = np.where(A==-np.inf,Z,A)
                if self.norm_lap_matr:
                    for i in range(A.shape[0]):
                        G = nx.from_numpy_matrix(A[i,:,:])
                        A[i,:,:] = nx.normalized_laplacian_matrix(G).toarray()
        elif self.adj_type == "l2_distance":
            V = seq.transpose(2,0,1)
            dist = V.unsqueeze(2) - V.unsqueeze(1)
            A = torch.sqrt((dist**2).sum(3))
        else:
            V=None
            A=None
        
        if need_A:
            return V,A
        else:
            return V,None
    
    def get_target_angle(self,rotate_type):
        idx = self.angle_hist["prob"].multinomial(num_samples=1, replacement=True)
        theta = self.angle_hist["angle"][int(idx)]
        return theta

    def get_target_scale(self):
        idx = self.scale_hist["prob"].multinomial(num_samples=1, replacement=True)
        gamma = self.scale_hist["angle"][int(idx)]
        return gamma
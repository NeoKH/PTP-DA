#!/usr/bin/env python3
# coding:utf8
"""
@file      eth_ucy_graph.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import math
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm

from .eth_ucy_base import ETH_UCY_BaseData
from utils.data import anorm

class ETH_UCY_GtaphData(ETH_UCY_BaseData):
    """ETH_UCY_GtaphData

    """

    def __init__(
        self,
        target_name:str,
        data_path: str, 
        len_observe: int = 8, 
        len_predict: int = 12, 
        skip: int = 1, 
        min_ped: int = 2, 
        threshold: float = 0.002, 
        delim: str = '\t', 
        relative_fun: str = "self_offset", 
        map_path:str=None,
        augment:list=None,
        rot_type:str="cluster",
        adj_type = "anorm",
        norm_lap_matr = True,
    ):
        super(ETH_UCY_GtaphData,self).__init__(
            target_name,data_path, len_observe, len_predict, skip, 
            min_ped, threshold, delim, relative_fun, map_path, augment,rot_type
        )
        
        self.adj_type = adj_type
        self.norm_lap_matr = norm_lap_matr
        
        # Convert to graph
        self.V_observes = []
        self.A_observes = [] 
        self.V_predicts = [] 
        self.A_predicts = []
        
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            
            start, end = self.seq_start_end[ss]

            v_,a_ = self.seq_to_graph(
                self.observes[start:end,:],
                self.observes_rel[start:end, :]
            )
            self.V_observes.append(v_.clone())
            self.A_observes.append(a_.clone())
            
            v_,a_ = self.seq_to_graph(
                self.predicts[start:end,:],
                self.predicts_rel[start:end, :]
            )
            self.V_predicts.append(v_.clone())
            self.A_predicts.append(a_.clone())
        pbar.close()
        
        
    def __getitem__(self, index):
        out = super(ETH_UCY_GtaphData,self).__getitem__(index)
        
        out["V_observe"] = self.V_observes[index],  # (T,N,2)
        out["A_observe"] = self.A_observes[index],  # (T,N,N)
        out["V_predict"] = self.V_predicts[index],  # (T,N,2)
        out["A_predict"] = self.A_predicts[index],  # (T,N,N)
        
        return out
    
    def seq_to_graph(self, seq, seq_rel):
        if self.adj_type == "anorm":
            seq_len = seq.shape[2]
            max_nodes = seq.shape[0]
            V = seq_rel.permute(2,0,1)
            I = torch.eye(max_nodes)
            Z = torch.zeros((seq_len,max_nodes,max_nodes))
            dist = V.unsqueeze(2) - V.unsqueeze(1)
            A = torch.sqrt((dist**2).sum(3))
            A = A + I
            A = 1/A
            A = torch.where(A==np.inf,Z,A)
            if self.norm_lap_matr:
                for i in range(A.shape[0]):
                    G = nx.from_numpy_matrix(A[i,:,:].numpy())
                    A[i,:,:] = torch.from_numpy(
                        nx.normalized_laplacian_matrix(G).toarray()
                        #! FutureWarning: normalized_laplacian_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.
                    ).type(torch.float)
            
        elif self.adj_type == "l2_distance":
            seq = seq.squeeze()
            seq_rel = seq_rel.squeeze()
            V = torch.from_numpy(np.array(seq_rel)).type(torch.float)
            V = V.permute(2,0,1)
            dist = V.unsqueeze(2) - V.unsqueeze(1)
            A = torch.sqrt((dist**2).sum(3))

        return V,A


if __name__=="__main__":
    eth = ETH_UCY_GtaphData(
        data_path="/home/kh/Datasets/ETH_UCY_OldSplit/eth/test/",
    )
    print(eth.__len__())


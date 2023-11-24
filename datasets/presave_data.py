#!/usr/bin/env python3
# coding:utf8
"""
@file      presave_data.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import torch
import random
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from torch.utils.data import Dataset
from datasets.unified_data import UnifiedData
from utils.data import angle_statistic,scale_statistic,get_hist
from utils.data import get_sequence_average_angle,get_sequence_angle

class Presave_Data(Dataset):
    def __init__(self,args,predata_path,predata_regenerate:bool=False) -> None:
        super(Presave_Data,self).__init__()
        self.predata_path = predata_path
        if predata_regenerate:
            self.data = self.preprocess(args)
        else:
            with open(self.predata_path,"rb") as f:
                self.data = pickle.load(f)
        # self.data = list(self.data)
    def __getitem__(self, index) -> dict:
        return self.data[index]
    
    def __len__(self,):
        return len(self.data)
    
    def preprocess(self,args):
        source_data = UnifiedData(
            data_root = args.data_root,
            archive_name= args.archive,
            data_name = args.source,
            data_type = "test", # 全部source data
            
            len_observe = args.len_observe,
            len_predict = args.len_predict,
            
            skip = args.overlap,
            min_ped = args.min_ped,
            relative_type = args.relative_type
        )
        self.adj_type = args.adj_type
        self.norm_lap_matr= args.not_norm_lap_matr
        def _package(_seq,_seq_rel):
            angle = get_sequence_angle(_seq_rel)
            observes = _seq[:,:,:args.len_observe]
            predicts = _seq[:,:,args.len_observe:]
            observes_rel = _seq_rel[:,:,:args.len_observe]
            predicts_rel = _seq_rel[:,:,args.len_observe:]
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
                "angle": angle,
            }
            return out

        angle_file_path = Path(args.angle_csv) / args.archive / args.target_type / f"{args.target}.pkl"
        scale_file_path = Path(args.scale_csv) / args.archive / args.target_type / f"{args.target}.pkl"
        if not angle_file_path.exists() or args.angle_regenerate :
            angle_statistic(
                src_root=args.data_root,
                dst_root=args.angle_csv,
                archive_name=args.archive,
                data_name=args.target,
                data_type = args.target_type,
            )
        if args.augment=="angle_target" or args.augment=="both_target":
            self.angle_hist = get_hist(angle_file_path,args.angle_bins,args.angle_obs_len)
            self.angle_hist["prob"] = torch.tensor(self.angle_hist["prob"])
        
        if not scale_file_path.exists() or args.scale_regenerate :
            scale_statistic(
                src_root=args.data_root,
                dst_root=args.scale_csv,
                archive_name=args.archive,
                data_name=args.target,
                data_type = args.target_type,
            )
        if args.augment=="scale_target"  or args.augment=="both_target":
            self.scale_hist = get_hist(scale_file_path,args.scale_bins,args.scale_obs_len)
            self.scale_hist["prob"] = torch.tensor(self.scale_hist["prob"])
        
        self.random_angles = list(range(-180,180,30))

        outs = []
        for index in range(len(source_data)):
            start, end = source_data.seq_start_end[index]
            
            seq = source_data.seq[start:end,:]
            seq_rel = source_data.seq_rel[start:end,:] # (N,2,T)
            outs.append(_package(seq,seq_rel))
            if args.augment=="none":continue # no augmentation
            if args.augment.find("both")!=-1:
                mode_dict = ["scale","angle","angle","angle","both"]
                for augment in mode_dict:
                    if augment=="both":
                        _augment = args.augment.replace(augment,"scale")
                        _seq, _seq_rel = self.augment_func(seq, seq_rel, _augment, args,"all")
                        _augment = args.augment.replace(augment,"angle")
                        _seq, _seq_rel = self.augment_func(_seq, _seq_rel, _augment, args,"all")
                    else:
                        _augment = args.augment.replace("both",augment)
                        _seq, _seq_rel = self.augment_func(seq, seq_rel, _augment, args,"all")
                    outs.append(_package(_seq,_seq_rel))
                pass
            else:
                for _ in range(4):
                    _seq, _seq_rel = self.augment_func(seq, seq_rel, args.augment, args,"all")
                    outs.append(_package(_seq,_seq_rel))

        if not self.predata_path.parent.exists():
            self.predata_path.parent.mkdir(parents=True)
        with open(self.predata_path,"wb") as f:
            pickle.dump(outs,f)
        return outs
        
    def augment_func(self,seq, seq_rel, augment, args, mode):
        seq, seq_rel = self.mode_select(seq,seq_rel,mode)
        if augment == "angle_target":
            source_angle = get_sequence_average_angle(seq_rel[:,:,1:args.len_observe])
            target_angle = self.get_target_angle()
            rotate_angle = target_angle - source_angle
            rotate_mat = np.asarray([
                        [np.cos(rotate_angle),-np.sin(rotate_angle)],
                        [np.sin(rotate_angle),np.cos(rotate_angle)]
            ],dtype=float)
            seq = np.matmul(rotate_mat,seq)
            seq_rel = np.matmul(rotate_mat,seq_rel)
            
        if augment == "angle_random":
            rotate_angle = random.sample(self.random_angles,1)[0]
            rotate_mat = np.asarray([
                        [np.cos(rotate_angle),-np.sin(rotate_angle)],
                        [np.sin(rotate_angle),np.cos(rotate_angle)]
            ],dtype=float)
            seq = np.matmul(rotate_mat,seq)
            seq_rel = np.matmul(rotate_mat,seq_rel)

        return seq,seq_rel

    def mode_select(self,seq,seq_rel,mode):
        if mode=="random_sample":
            n = seq.shape[0]
            idx_list = list(range(n))
            random.shuffle(idx_list)
            m = random.randint(1,n-1)
            new_seq = np.zeros((m,seq.shape[1],seq.shape[2]))
            new_seq_rel = np.zeros((m,seq_rel.shape[1],seq_rel.shape[2]))
            for i in range(m):
                new_seq[i,:] = seq[idx_list[i],:]
                new_seq_rel[i,:] = seq_rel[idx_list[i],:]
            return new_seq, new_seq_rel
        if mode == "all":
            return seq, seq_rel
        if mode == "same_direction":
            m = random.randint(0,seq.shape[0]-1)
            new_seq = [seq_rel[m,:][None,:]]
            new_seq_rel = [seq_rel[m,:][None,:]]

            _seq_rel = seq_rel[m,:,:8] # 2x8
            dis = (_seq_rel[:,1:]- _seq_rel[:,:-1]).T
            dis_wo_still = np.array([x for x in dis if np.any(x!=0) ])
            if dis_wo_still.size==0: return seq, seq_rel
            angle1 = np.mean(np.arctan2(dis_wo_still[:,1],dis_wo_still[:,0]))
            for i in range(seq.shape[0]):
                if i==m:continue
                _seq_rel = seq_rel[i,:,:8] # 2x8
                dis = (_seq_rel[:,1:]- _seq_rel[:,:-1]).T
                angle2 = np.mean(np.arctan2(dis[:,1],dis[:,1]))
                if np.abs(np.rad2deg(angle1-angle2))< 20:
                    new_seq.append(seq[i,:][None,:])
                    new_seq_rel.append(seq_rel[i,:][None,:])
            new_seq = np.concatenate(new_seq,axis=0)
            new_seq_rel = np.concatenate(new_seq_rel,axis=0)
            return new_seq,new_seq_rel
        
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
    
    def get_target_angle(self):
        idx = self.angle_hist["prob"].multinomial(num_samples=1, replacement=True)
        theta = self.angle_hist["angle"][int(idx)]
        return theta
    
    def get_target_scale(self):
        idx = self.scale_hist["prob"].multinomial(num_samples=1, replacement=True)
        gamma = self.scale_hist["angle"][int(idx)]
        return gamma
    

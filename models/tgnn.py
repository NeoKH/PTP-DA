#!/usr/bin/env python3
# coding:utf8
"""
@file      tgnn.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class TransferableGNN(nn.Module):
    def __init__(self,
                 feature_dims = 64,
                 output_size = 5,
                 len_obs = 8,
                 len_pred = 12,
                 num_gcn = 3,
                 num_tcn = 3,
                 max_peds = 100,
                 device = "cuda",
                 ):
        super(TransferableGNN,self).__init__()
        
        self.num_gcn = num_gcn
        self.num_tcn = num_tcn
        self.feature_dims = feature_dims
        self.output_size = output_size
        self.len_obs = len_obs
        self.len_pred = len_pred
        
        self.node_encode = nn.Sequential(
            nn.Linear(2,feature_dims),
            nn.ReLU(inplace=True),
        )
        
        self.st_gcns = nn.ModuleList()
        for i in range(num_gcn):
            self.st_gcns.append(st_gat(feature_dims = feature_dims, max_peds=max_peds, device=device))
        
        # TODO: W_f 具体取多大规模可以进行试验
        self.dv = feature_dims * len_obs # 512
        # TODO: share weights or not
        self.beta = nn.Sequential(
            nn.Linear(self.dv,self.dv//2,bias=False), # TODO
            nn.Tanh(),
            nn.Linear(self.dv//2,1,bias=False),
            nn.Softmax(dim=-2)
        )
        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(self.len_obs,self.len_pred,3,padding=1))
        for j in range(1,self.num_tcn):
            self.tpcnns.append(nn.Conv2d(self.len_pred,self.len_pred,3,padding=1))
        self.tpcnn_ouput = nn.Sequential(
            nn.Linear(self.feature_dims,self.output_size,bias=False),
            # nn.PReLU()
        )
        
        
        self.prelus = nn.ModuleList()
        for j in range(self.num_tcn):
            self.prelus.append(nn.PReLU())
        
    def forward(self,V_s,A_s,V_t,A_t):
        # print(V_s.shape)
        # encode source features
        F_s = self.node_encode(V_s)
        for k in range(1,self.num_gcn):
            F_s,A_s = self.st_gcns[k](F_s,A_s)
        
        # F_s: (1,8,N,Feature_dims)
        V_temp = torch.clone(F_s).view(F_s.shape[0],F_s.shape[1],F_s.shape[3],F_s.shape[2])
        # print(V_temp.shape)
        # V_temp: (1,8,Feature_dims,N)

        
        F_s = F_s.permute(0,2,3,1) # (1,N,Feature_dims,8)
        F_s = F_s.reshape(F_s.shape[0],-1,self.dv)
        beta_s = self.beta(F_s)
        C_s = torch.sum(beta_s * F_s,dim=1)
        
        # encode target features
        F_t = self.node_encode(V_t)
        for k in range(0,self.num_gcn):
            F_t,A_t = self.st_gcns[k](F_t,A_t)
        F_t = F_t.permute(0,2,3,1)
        F_t = F_t.reshape(F_t.shape[0],-1,self.dv)
        beta_t = self.beta(F_t)
        C_t = torch.sum(beta_t * F_t,dim=1)
        
        # decode source features
        # according the paper, we used social-STGCNN decoder
        V_temp = self.prelus[0](self.tpcnns[0](V_temp))

        for k in range(1,self.num_tcn-1):
            V_temp =  self.prelus[k](self.tpcnns[k](V_temp)) + V_temp
        
        # print(V_temp.shape)
        V_temp = V_temp.view(V_temp.shape[0],V_temp.shape[1],V_temp.shape[3],V_temp.shape[2])
        V_temp = self.tpcnn_ouput(V_temp)
        
        # V_temp = torch.clamp(V_temp,min=-10,max=10)
        
        # if torch.sum(torch.isnan(F_s)):
        #     print("F_s",F_s)
        # if torch.sum(torch.isnan(A_s)):
        #     print("A_s",A_s)
        # if torch.sum(torch.isnan(beta_s)):
        #     print("beta_s",beta_s)
        
        return C_s,C_t,V_temp

class JustGNN(nn.Module):
    def __init__(self,
                 feature_dims = 64,
                 output_size = 5,
                 len_obs = 8,
                 len_pred = 12,
                 num_gcn = 3,
                 num_tcn = 3,
                 max_peds = 100,
                 device = "cuda:0",
                 ):
        super(JustGNN,self).__init__()
        
        self.num_gcn = num_gcn
        self.num_tcn = num_tcn
        self.feature_dims = feature_dims
        self.output_size = output_size
        self.len_obs = len_obs
        self.len_pred = len_pred
        
        self.node_encode = nn.Sequential(
            nn.Linear(2,feature_dims),
            nn.ReLU(inplace=True),
        )
        
        self.st_gcns = nn.ModuleList()
        for i in range(num_gcn):
            self.st_gcns.append(st_gat(feature_dims = feature_dims, max_peds=max_peds, device=device))
        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(self.len_obs,self.len_pred,3,padding=1))
        for j in range(1,self.num_tcn):
            self.tpcnns.append(nn.Conv2d(self.len_pred,self.len_pred,3,padding=1))
        self.tpcnn_ouput = nn.Sequential(
            nn.Linear(self.feature_dims,self.output_size,bias=False),
            # nn.PReLU()
        )

        self.prelus = nn.ModuleList()
        for j in range(self.num_tcn):
            self.prelus.append(nn.PReLU())
        
    def forward(self,V_s,A_s):
        # print(V_s.shape)
        # encode source features
        F_s = self.node_encode(V_s)
        for k in range(1,self.num_gcn):
            F_s,A_s = self.st_gcns[k](F_s,A_s)
        
        V_temp = torch.clone(F_s).view(F_s.shape[0],F_s.shape[1],F_s.shape[3],F_s.shape[2])
        # print(V_temp.shape)
        # V_temp: (Batch_size,L_obs,Feature_dims,num_peds
        
        # decode source features
        # according the paper, we used social-STGCNN decoder
        V_temp = self.prelus[0](self.tpcnns[0](V_temp))

        for k in range(1,self.num_tcn-1):
            V_temp =  self.prelus[k](self.tpcnns[k](V_temp)) + V_temp
        
        # print(V_temp.shape)
        V_temp = V_temp.view(V_temp.shape[0],V_temp.shape[1],V_temp.shape[3],V_temp.shape[2])
        V_temp = self.tpcnn_ouput(V_temp)
        
        return V_temp


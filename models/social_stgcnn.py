#!/usr/bin/env python3
# coding:utf8
"""
@file      social_stgcnn.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A) #RuntimeError: einsum(): the number of subscripts in the equation (3) does not match the number of dimensions (1) for operand 1 and no ellipsis was given

        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

class social_stgcnn(nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(social_stgcnn,self).__init__()
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn
                
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))
        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)
            
            
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())


    def forward(self,v,a):

        for k in range(self.n_stgcnn):
            v,a = self.st_gcns[k](v,a)
            
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        
        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1,self.n_txpcnn-1):
            v =  self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)

        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])

        return v,a

class SocialSTGCNN_Adaptive(social_stgcnn):
    def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=2, output_feat=5, 
                 seq_len=8, pred_seq_len=12, kernel_size=3):
        super().__init__(n_stgcnn, n_txpcnn, input_feat, output_feat, 
                         seq_len, pred_seq_len, kernel_size)
        
        self.dv = seq_len * output_feat
        
        self.adaptiver = nn.Sequential(
            nn.Linear(self.dv,self.dv,bias=False),
            nn.Tanh(),
            nn.Linear(self.dv,1,bias=False),
            nn.Softmax(dim=-2),
        )
        
    
    def forward(self,v_s,a_s,v_t=None,a_t=None):
        # v_s: torch.Size([1, 2, 8, N])
        # a_s: torch.Size([1, 8, N, N])
        
        # Source Encoder
        for k in range(self.n_stgcnn):
            v_s,a_s = self.st_gcns[k](v_s,a_s)
        
        v_s_cp = torch.clone(v_s)
        
        # v_s: torch.Size([1, 5, 8, N])
        v_s = v_s.view(v_s.shape[0],v_s.shape[2],v_s.shape[1],v_s.shape[3])
        # v_s: torch.Size([1, 8, 5, N])
        
        # Source Decoder
        v_s = self.prelus[0](self.tpcnns[0](v_s)) # v_s: torch.Size([1, 12, 5, N])
        
        for k in range(1,self.n_txpcnn-1):
            v_s =  self.prelus[k](self.tpcnns[k](v_s)) + v_s
        
        v_s = self.tpcnn_ouput(v_s) # torch.Size([1, 12, 5, N])
        
        v_s = v_s.view(v_s.shape[0],v_s.shape[2],v_s.shape[1],v_s.shape[3])
        # torch.Size([1, 5, 12, N])
        
        if v_t is None:
            return v_s, None,None
        else:
            # Target Encoder
            for k in range(self.n_stgcnn):
                v_t,a_t = self.st_gcns[k](v_t,a_t)
            
            # Source to target
            v_s_cp = v_s_cp.permute(0,3,1,2).reshape(1,-1,self.dv)
            v_t = v_t.permute(0,3,1,2).reshape(1,-1,self.dv)
            c_s = torch.sum(self.adaptiver(v_s_cp)*v_s_cp,dim=1)
            c_t = torch.sum(self.adaptiver(v_t)*v_t,dim=1)
            
            # print(c_s.shape) # torch.Size([1, 40])
            # print(c_t.shape) # torch.Size([1, 40])
        
            return v_s, c_s, c_t


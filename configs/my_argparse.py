#!/usr/bin/env python3
# coding:utf8
"""
@file      my_argparse.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import argparse

class BaseArgparse(object):
    
    def __init__(self,) -> None:
        self.parser = argparse.ArgumentParser()

        #Data specifc paremeters
        self.parser.add_argument('--len_observe', type=int, default=8)
        self.parser.add_argument('--len_predict', type=int, default=12)
        self.parser.add_argument('--min_ped', type=int, default=2)
        self.parser.add_argument('--overlap', type=int, default=1)
        self.parser.add_argument('--data', default='eth',help='when DA,it is not uesd')
        self.parser.add_argument('--archive', default='ETH_UCY_DA')
        self.parser.add_argument('--data_root', default='./data/origin',help='datasets parent path')
        self.parser.add_argument('--adj_type', type=str, default= 'anorm' )
        self.parser.add_argument('--relative_type', type=str, default= 'self_offset' )
        self.parser.add_argument('--not_norm_lap_matr', action="store_false" )
        
        # Angle augmentation
        self.parser.add_argument('--angle_bins', type=int, default= 45 )
        self.parser.add_argument('--angle_prob', type=float, default= 0.8)
        self.parser.add_argument('--angle_obs_len', type=int, default= 8 )
        self.parser.add_argument('--angle_regenerate', action="store_true", default=False,help='regenerate target angles or not')
        self.parser.add_argument('--angle_csv', type=str, default='./data/statistic/angle')
        # Scale augmentation
        self.parser.add_argument('--scale_bins', type=int, default= 45 )
        self.parser.add_argument('--scale_prob', type=float, default= 0.8)
        self.parser.add_argument('--scale_obs_len', type=int, default= 8 )
        self.parser.add_argument('--scale_regenerate', action="store_true", default=False,help='regenerate target angles or not')
        self.parser.add_argument('--scale_csv', type=str, default='./data/statistic/scale')
        # Offline preprocess
        # self.parser.add_argument('--use_preprocess', action="store_true", default=False,help='Use preprocess data')
        self.parser.add_argument('--preprocess_path', default='./data/preprocess',help='datasets parent path')
        self.parser.add_argument('--predata_regenerate', action="store_true", default=False,help=' ')
        self.parser.add_argument('--online', action="store_true",default=False,help='online or offline')

        # ckpt config
        self.parser.add_argument('--save_ckpt',action="store_true",default=True,help="save ckpt or not")
        self.parser.add_argument('--ckpt_period', type=int, default= 100 )
        
        #Training specifc parameters
        self.parser.add_argument('--device', type=str, default="cuda:0",help='gpu or cpu')
        self.parser.add_argument('--visual_mode', type=str, default="visualdl",help='tensorboard or visualdl or both')
        self.parser.add_argument('--batch_size', type=int, default=128,help='minibatch size')
        self.parser.add_argument('--num_workers', type=int, default=2,help='num_workers')
        
        self.parser.add_argument('--resume', action="store_true", default=False,help='Use pretrained model')
        self.parser.add_argument('--finetune', action="store_true", default=False,help='Use pretrained model')
        self.parser.add_argument('--pretrained_path', default='./runs/stgcnn/',help=' path')
        
        self.parser.add_argument('--num_epochs', type=int, default=300,help='number of epochs')  
        self.parser.add_argument('--start_epoch', type=int, default=0,help='number of epochs')  
        
        self.parser.add_argument('--clip_grad', type=float, default=None, help='gadient clipping')
        
        self.parser.add_argument('--lr', type=float, default=0.01,help='learning rate')
        self.parser.add_argument('--lr_sh_rate', type=int, default=150,help='number of steps to drop the lr')
        self.parser.add_argument('--use_lrschd', action="store_true", default=True,help='Use lr rate scheduler')
        self.parser.add_argument('--lrschd_mode', type=str,default="step") # "cosine"
        self.parser.add_argument('--warmup', type=int, default=5,help='the number of warmup lasts')
        self.parser.add_argument('--lr_min', type=float, default=1e-4,help='')
        self.parser.add_argument('--warmup_lr_init', type=float, default=1e-3,help='')
        self.parser.add_argument('--weight_decay', type=float, default=1e-3,help='')
        
        self.parser.add_argument('--optimizer', type=str,default="sgd") # "adam"
        
        self.parser.add_argument('--patience', type=int, default=70,help='early stopping')

        # Test parameters
        self.parser.add_argument('--test_all', action="store_true", default=False)
        self.parser.add_argument('--num_samples', type=int,default=20)
        self.parser.add_argument('--model_path', type=str,default="./runs")
        
        self.parser.add_argument('--ablation_mode', type=str,default="") # "scale angle x"

    def parse_args(self,):
        return self.parser.parse_args()
    
class STGCNN_parser(BaseArgparse):
    def __init__(self) -> None:
        super().__init__()

        #Model specific parameters
        self.parser.add_argument('--input_size', type=int, default=2)
        self.parser.add_argument('--output_size', type=int, default=5)
        self.parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
        self.parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
        self.parser.add_argument('--kernel_size', type=int, default=3)
        
        self.parser.add_argument('--data_split_mode', type=str, default="tgnn")
        
class STGCNN_Adaptive_parser(STGCNN_parser):
    def __init__(self) -> None:
        super().__init__()
        self.parser.add_argument("--model_name",default="stgcnn",type=str)
        self.parser.add_argument("--model_type",default="source_only",type=str)
        # self.parser.add_argument('--augments', type=str,nargs='+',default="",help="target_angle,random_angle or target_scale")
        self.parser.add_argument('--augment', type=str,default="none",help="angle_target,scale_target or both_scale")
        #Data specifc paremeters
        self.parser.add_argument('--source', default='eth',help='eth,hotel,univ,zara1 zara2')    
        self.parser.add_argument('--target', default='hotel',help='eth,hotel,univ,zara1 zara2')
        self.parser.add_argument('--alpha', type=float,default=1.0,help='trade off two loss')

        self.parser.add_argument('--fine_augment', type=str,default="none",help="angle_target,scale_target or both_scale")
        self.parser.add_argument("--fine_model_type",default="source_only",type=str)

# class Preprocess_parser(BaseArgparse):

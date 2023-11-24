#!/usr/bin/env python3
# coding:utf8
"""
@file      test.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import sys
import copy
from datetime import datetime
import glob

import torch
import pickle
import numpy as np
from pathlib import Path
from natsort import ns, natsorted
from torch.utils.data import DataLoader
import torch.distributions.multivariate_normal as torchdist

from configs.my_argparse import STGCNN_Adaptive_parser
from models import SocialSTGCNN_Adaptive
from datasets.unified_graph import UnifiedGraph
from utils.generals import same_seeds
from utils.log import StdoutLogger
from utils.metrics import ade,fde,seq_to_nodes,nodes_rel_to_nodes_abs

def test_one(args):
    same_seeds(42)
    # path check
    model_path = Path(args.model_path) / args.model_name / args.model_type \
        / args.augment /f"{args.source}2{args.target}"
    assert model_path.exists()
    # results file
    sys.stdout = StdoutLogger(str(model_path / f"{args.source}2{args.target}.txt"))
    
    print("="*70)
    print(f"Model: {str(model_path)}")
    print("Time: ", datetime.now())
    
    model_p = model_path / "val_best.pth"
    args_p = model_path / "args.pkl"
    with open(str(args_p),'rb') as f:
        train_args = pickle.load(f)
    
    best_metric_p = model_path / "best_metric.pkl"
    if best_metric_p.exists():
        with open(str(best_metric_p),'rb') as f:
            best_metric = pickle.load(f)
        print(f"best_val_epoch: {best_metric['best_val_epoch']}, best_val_loss: {best_metric['best_val_loss']}")
    
    print(f"Source : {args.source}\t Target :{args.target}")

    if args.data_split_mode == "ours":
        val_type = "val"
    elif args.data_split_mode == "tgnn":
        val_type = "train"
    else:
        val_type = "train"
    
    test_data = UnifiedGraph(
        data_root = args.data_root,
        archive_name=args.archive,
        data_name = args.target,
        data_type = val_type ,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle =False,
        num_workers=args.num_workers
    )
    
    #Defining the model
    model = SocialSTGCNN_Adaptive(
        n_stgcnn    = train_args.n_stgcnn,
        n_txpcnn    = train_args.n_txpcnn,
        output_feat = train_args.output_size,
        seq_len     = train_args.len_observe,
        pred_seq_len= train_args.len_predict,
        kernel_size = train_args.kernel_size,
    ).to(args.device)
    model.load_state_dict(torch.load(model_p))
    
    ad_20,fd_20,ad_1,fd_1,raw_data_dic_= test(model,test_loader, args)
    print("Num samples: ",args.num_samples)
    print(f"ADE:\t{ad_20:.6f}\t{ad_20:.2f}")
    print(f"FDE:\t{fd_20:.6f}\t{fd_20:.2f}\n")
    print("Num samples: 1")
    print(f"ADE:\t{ad_1:.6f}\t{ad_1:.2f}")
    print(f"FDE:\t{fd_1:.6f}\t{fd_1:.2f}\n")

def test(model,loader,args):
    model.eval()
    ade_20_bigls = []
    fde_20_bigls = []
    ade_1_bigls = []
    fde_1_bigls = []
    raw_data_dict = {}
    step =0 

    for batch in loader: 
        step+=1
        #Get data

        obs_traj = batch["observe"][0].to(args.device).unsqueeze(0)
        pred_traj_gt = batch["predict"][0].to(args.device).unsqueeze(0)
        obs_traj_rel = batch["observe_rel"][0].to(args.device).unsqueeze(0) # 1xNx2x8

        V_obs = batch["V_observe"].to(args.device).permute(0,3,1,2) # batch,feat,seq,node
        A_obs = batch["A_observe"].to(args.device).squeeze(0)
        V_gt  = batch["V_predict"].to(args.device).squeeze(0)

        #Forward
        V_pred,_,_ = model(V_obs,A_obs)
        # print(V_pred.shape) # torch.Size([1, 5, 12, N])
        V_pred = V_pred.permute(0,2,3,1).squeeze(0) # torch.Size([1, 12, N, 5])>>seq,node,feat
        V_obs  = V_obs.permute(0,2,3,1).squeeze(0) # torch.Size([1, 8, N, 2])>>seq,node,feat

        num_of_objs = obs_traj_rel.shape[1]
        V_pred =  V_pred[:,:num_of_objs,:]
        V_gt   =  V_gt[:,:num_of_objs,:]

        #For now I have my bi-variate parameters 
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).to(args.device)
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]

        mvnormal = torchdist.MultivariateNormal(mean,cov)

        ### Rel to abs 
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 
        
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().copy(),
                                                 V_x[0,:,:].copy())
        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_gt.data.cpu().numpy().copy(),
                                                 V_x[-1,:,:].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]
        
        # sample 20
        for k in range(args.num_samples):
            V_pred = mvnormal.sample()

            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().copy(),
                                                     V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
            # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = [] 
                target = []
                obsrvs = [] 
                number_of = []
                pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                target.append(V_y_rel_to_abs[:,n:n+1,:])
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)

                ade_ls[n].append(ade(pred,target,number_of))
                fde_ls[n].append(fde(pred,target,number_of))
        
        for n in range(num_of_objs):
            ade_20_bigls.append(min(ade_ls[n]))
            fde_20_bigls.append(min(fde_ls[n]))
        for n in range(num_of_objs):
            ade_1_bigls.append(np.mean(ade_ls[n]))
            fde_1_bigls.append(np.mean(fde_ls[n]))
        
    ade_20 = sum(ade_20_bigls)/len(ade_20_bigls)
    fde_20 = sum(fde_20_bigls)/len(fde_20_bigls)
    ade_1 = sum(ade_1_bigls)/len(ade_1_bigls)
    fde_1 = sum(fde_1_bigls)/len(fde_1_bigls)
    
    return ade_20,fde_20,ade_1,fde_1,raw_data_dict

def results_gather(
    root = "./runs/",
    model_name = "tgnn",
    model_type="source_only",
    augment_name="angle_target",
):
    
    file_path = Path(root) / model_name / model_type / augment_name
    assert file_path.exists()
    p_files = []
    for dir in natsorted(file_path.iterdir(),alg=ns.PATH):
        if dir.is_dir():
            for x in dir.iterdir():
                if x.suffix==".txt" and x.stem!="stdout":
                    p_files.append(x)
    all_txt = ""
    for p_file in p_files:
        s = p_file.read_text()
        all_txt +=s
    
    save_path = file_path / "all.txt"
    save_path.write_text(all_txt) 

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    parser = STGCNN_Adaptive_parser()
    args = parser.parse_args()
    if not args.test_all:
        test_one(args)
    else:
        results_gather(
            model_type = args.model_type,
            augment_name = args.augment,
        )


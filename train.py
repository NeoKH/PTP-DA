#!/usr/bin/env python3
# coding:utf8
"""
@file      train.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import sys
from datetime import datetime
import random

import torch
import pickle
import numpy as np
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.my_argparse import STGCNN_Adaptive_parser
from models import SocialSTGCNN_Adaptive
from datasets.unified_graph import UnifiedGraph
from datasets.presave_data import Presave_Data
from utils.generals import same_seeds
from utils.log import StdoutLogger,VisualLogger
from utils.metrics import total_loss , bivariate_loss
from utils.data import kde_compare
from test import test_one

def main(args):
    same_seeds(42)
    
    save_path = Path(args.model_path) / args.model_name / args.model_type \
        / args.augment /f"{args.source}2{args.target}"
    ckpt_path = save_path / "ckpts"
    if not ckpt_path.exists():
        ckpt_path.mkdir(parents = True)
    
    # logger
    sys.stdout = StdoutLogger(str(save_path / "stdout.txt"))
    logger = VisualLogger(logdir=str(save_path),mode=args.visual_mode)
    print(f"Model:{str(save_path)}")

    print(f"Loading source: {args.source} dataset !")
    # 
    if args.data_split_mode == "ours":
        val_type = "val"
        args.target_type = "train"
    elif args.data_split_mode == "tgnn":
        val_type = "train"
        args.target_type = "val"
    else:
        val_type = "train"
        args.target_type = "val"
    if not args.online:
        if save_path.parts[-2].find("target")!=-1:
            predata_path = Path(args.preprocess_path) / args.data_split_mode / save_path.parts[-2] /f"{args.source}2{args.target}.pkl"
        else:
            predata_path = Path(args.preprocess_path) / args.data_split_mode / save_path.parts[-2] /f"{args.source}.pkl"
        if not predata_path.exists() or args.predata_regenerate:
            source_data = Presave_Data(args,predata_path,True)
        else:
            source_data= Presave_Data(args,predata_path,False)
    else:
        source_data = UnifiedGraph(
            data_root = args.data_root,
            archive_name= args.archive,
            data_name = args.source,
            data_type = "test",
            
            len_observe = args.len_observe,
            len_predict = args.len_predict,
            
            skip = args.overlap,
            min_ped = args.min_ped,
            adj_type = args.adj_type,
            relative_type = args.relative_type,
            norm_lap_matr= args.not_norm_lap_matr,
            
            augment = args.augment,
            target_name=args.target, # work only in target mode
            target_type = args.target_type,
            # angle augmentation
            angle_bins = args.angle_bins,
            angle_prob = args.angle_prob,
            angle_obs_len = args.angle_obs_len,
            angle_regenerate = args.angle_regenerate,
            angle_csv = args.angle_csv,
            
            # scale augmentation
            scale_bins = args.scale_bins,
            scale_prob = args.scale_prob,
            scale_obs_len = args.scale_obs_len,
            scale_regenerate = args.scale_regenerate,
            scale_csv = args.scale_csv,
        )
    # return
    source_loader = DataLoader(
        source_data,batch_size=1,shuffle =True,num_workers=args.num_workers
    )
    print(f"Loading the source dataset is complete !")

    print(f"Loading target:{args.target} validation set !")
    val_data = UnifiedGraph(
        data_root = args.data_root,
        archive_name=args.archive,
        data_name = args.target,
        data_type = val_type,
        
        len_observe = args.len_observe,
        len_predict = args.len_predict,
        
        skip = args.overlap,
        min_ped = args.min_ped,
        adj_type = args.adj_type,
        relative_type = args.relative_type,
        norm_lap_matr= args.not_norm_lap_matr,
    )
    val_loader = DataLoader(
        val_data,batch_size=1,
        shuffle =False,num_workers=args.num_workers
    )
    print(f"Loading target validation set is complete !")

    if args.model_type.find("adaptation")!=-1:
        print(f"Loading target:{args.target} train set !")
        target_data = UnifiedGraph(
            data_root = args.data_root,
            archive_name=args.archive,
            data_name = args.target,
            data_type = args.target_type,
            
            len_observe = args.len_observe,
            len_predict = args.len_predict,
            
            skip = args.overlap,
            min_ped = args.min_ped,
            adj_type = args.adj_type,
            relative_type = args.relative_type,
            norm_lap_matr= args.not_norm_lap_matr,
        )
        print(f"Loading target train set is complete !")
    else:
        target_data = None

    # Define the model 
    model = SocialSTGCNN_Adaptive(
        n_stgcnn    = args.n_stgcnn,
        n_txpcnn    = args.n_txpcnn,
        output_feat = args.output_size,
        seq_len     = args.len_observe,
        pred_seq_len= args.len_predict,
        kernel_size = args.kernel_size,
    ).to(args.device)
    
    #Training settings 
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr = args.lr,
            # weight_decay = args.weight_decay # 1e-3
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr = args.lr,
            # weight_decay = args.weight_decay # 1e-3
        )
    if args.use_lrschd:
        if args.lrschd_mode.lower().find("step")!=-1:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
        elif args.lrschd_mode.lower().find("cos")!=-1:
            import timm
            import timm.optim
            import timm.scheduler
            scheduler = timm.scheduler.CosineLRScheduler(
                optimizer,
                t_initial=args.num_epochs,
                lr_min=args.lr_min,
                warmup_t=args.warmup,
                warmup_lr_init=args.warmup_lr_init,
            )
        else:
            args.use_lrschd = False
    else:
        scheduler = None
        
    with open(str(save_path/'args.pkl'), 'wb') as fp:
        pickle.dump(args, fp)
    
    # load pretrained model
    if args.resume:
        path_checkpoint = Path(args.pretrained_path) / args.model_type / args.augment / f"{args.source}2{args.target}" / "ckpts" / f"{args.start_epoch:03d}.pt"
        checkpoint = torch.load(str(path_checkpoint)) 
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        args.start_epoch = checkpoint['epoch'] 
    if args.finetune:
        path_checkpoint = Path(args.pretrained_path) / args.fine_model_type / args.fine_augment / f"{args.source}2{args.target}" / "val_best.pth"
        model.load_state_dict(torch.load(str(path_checkpoint)))

    start_time = datetime.now()
    print("Start time: ",start_time)
    best_metric =  {'best_val_epoch':-1, 'best_val_loss':1e9}
    pre_best_metric =  {'best_val_epoch':-1, 'best_val_loss':1e9}
    epochs_without_improvement = 0
    for epoch in range(args.start_epoch,args.num_epochs):
        train(epoch,model,optimizer,source_loader,target_data,logger,args)
        val(epoch,model,best_metric,val_loader,logger,save_path,args)
        
        if args.use_lrschd:
            scheduler.step(epoch)
            print(f"learning rate: {optimizer.param_groups[0]['lr']}")
            logger.add_scalar(tag="lr",scalar_value = optimizer.param_groups[0]['lr'],global_step = epoch)

        if epoch==args.start_epoch:
            cur_time = datetime.now()
            delta_time = cur_time - start_time
            print("*"*50)
            print(f"One epoch time consume:\t{delta_time}")
            print(f"Remaining time:\t{delta_time*(args.num_epochs-args.start_epoch-1)}")
            print("*"*50)

        if args.save_ckpt:
            if (epoch+1)%args.ckpt_period==0:
                # save ckpt
                num_params = sum(p.numel() for p in model.parameters())
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_metric['best_val_loss'], 
                    'best_val_epoch': best_metric['best_val_epoch'], 
                    'num_params': num_params
                }
                torch.save(checkpoint, str(save_path / "ckpts" /f"{epoch+1:03d}.pt"))

        # error detect
        if best_metric['best_val_epoch']==-1:
            print("##"*30)
            print("something wrong !")
            print("##"*30)
            break
        
        # early stopping
        if best_metric["best_val_loss"] < pre_best_metric["best_val_loss"]:
            epochs_without_improvement = 0
            pre_best_metric["best_val_loss"] = best_metric["best_val_loss"]
            pre_best_metric["best_val_epoch"] = best_metric["best_val_epoch"]
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement == args.patience:
            print('Early stopping at epoch {}...'.format(epoch+1))
            break

    with open(save_path/'best_metric.pkl', 'wb') as fp:
        pickle.dump(best_metric, fp)
    print("="*40)
    
    test_one(args)
    
    ps_angles = save_path/'source_angles.pkl'
    _angles = {8:angles}
    with open(ps_angles, 'wb') as fp:
        pickle.dump(_angles, fp)

    ps_scales = save_path/'source_scales.pkl'
    _scales = {8:scales}
    with open(ps_scales, 'wb') as fp:
        pickle.dump(_scales, fp)

    pt_angles = Path(args.angle_csv) / args.archive / args.target_type / f"{args.target}.pkl"
    kde_compare(
        ps_angles,pt_angles,
        args.source,args.target,
        pimg = str(save_path),dtype="angle",
        angle_obs_len = args.angle_obs_len,
        bins = args.angle_bins
    )
    pt_scales = Path(args.scale_csv) / args.archive / args.target_type / f"{args.target}.pkl"
    kde_compare(
        ps_scales,pt_scales,
        args.source,args.target,
        pimg = str(save_path),dtype="scale",
        angle_obs_len = args.scale_obs_len,
        bins = args.scale_bins
    )
    
    print("="*40)
    logger.close()
    end_time = datetime.now()
    print("End Time:",end_time)
    print("Total Spend:",end_time - start_time)

def train(epoch,model,optimizer,source_loader,target_data,logger,args):
    model.train()

    losses_batch = np.zeros(3,dtype=float) # [total_loss,traj_loss,adaptive_loss]
    loss_batch = 0
    
    source_len = len(source_loader)
    turn_point = source_len//args.batch_size*args.batch_size + source_len%args.batch_size -1
    batch_count = 0
    is_fst_loss = True
    for cnt,source_batch in enumerate(source_loader):
        
        batch_count+=1
        #Get source data
        s_V_obs = source_batch["V_observe"].to(args.device)
        s_A_obs = source_batch["A_observe"].to(args.device)
        s_V_gt  = source_batch["V_predict"].to(args.device)
        
        angles.extend([x.item() for x in source_batch["angle"]])
        scales.extend([x.item() for x in source_batch["scale"]])
        
        s_A_obs = s_A_obs.squeeze(0)
        s_V_obs = s_V_obs.permute(0,3,1,2)
        
        # Get target data
        if args.model_type.find("adaptation")!=-1:
            _idx = int(random.random()*len(target_data))
            t_V_obs = target_data[_idx]["V_observe"].to(args.device)
            t_A_obs = target_data[_idx]["A_observe"].to(args.device)

            t_V_obs = t_V_obs.unsqueeze(0)
            t_V_obs = t_V_obs.permute(0,3,1,2)

        # forward
        optimizer.zero_grad()
        if args.model_type.find("adaptation")!=-1:
            s_V_pred,s_features,t_features = model(s_V_obs,s_A_obs,t_V_obs,t_A_obs)
        elif args.model_type.find("source_only")!=-1:
            s_V_pred,s_features,t_features = model(s_V_obs,s_A_obs,None,None)
        s_V_pred = s_V_pred.permute(0,2,3,1).squeeze(0)
        s_V_gt = s_V_gt.squeeze(0)
        if args.model_type.find("adaptation")!=-1:
            s_features = s_features.squeeze(0)
            t_features = t_features.squeeze(0)
        
        if args.model_type.find("adaptation")!=-1:
            if batch_count % args.batch_size !=0 and cnt != turn_point :
                losses = total_loss(s_V_pred,s_V_gt,s_features,t_features,args.alpha)
                if is_fst_loss :
                    loss_single = losses
                    is_fst_loss = False
                else:
                    loss_single = [ x + y for x,y in zip(losses,loss_single)]
            else:
                real_batch = args.batch_size if batch_count % args.batch_size ==0 else cnt%args.batch_size
                # Metrics
                losses_batch += [x.item() for x in loss_single]
                
                loss_single = [ x/real_batch for x in loss_single]
                is_fst_loss = True
                loss_single[0].backward()
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
                optimizer.step()
                
                mean_losses  = losses_batch / batch_count
                
                # log
                logstr = f"TRAIN: {args.source}2{args.target} \t Epoch:{epoch:03d}  Batch:{batch_count:04d}"
                logstr += f"\t Pred_Loss:{mean_losses[1]:.6f}"
                logstr += f"\t Align_Loss:{mean_losses[2]:.6f}"
                logstr += f"\t Total_Loss:{mean_losses[0]:.6f}"
                print(logstr)
                logger.add_scalar(tag="train/loss",scalar_value = mean_losses[0],global_step = (epoch+1)*batch_count)
                logger.add_scalar(tag="train/pred_loss",scalar_value = mean_losses[1],global_step = (epoch+1)*batch_count)
                logger.add_scalar(tag="train/align_loss",scalar_value = mean_losses[2],global_step = (epoch+1)*batch_count)
        
        elif args.model_type.find("source_only")!=-1:
            if batch_count % args.batch_size !=0 and cnt != turn_point :
                l = bivariate_loss(s_V_pred,s_V_gt)
                if is_fst_loss :
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l
            else:
                real_batch =  args.batch_size if batch_count % args.batch_size ==0 else cnt%args.batch_size
                #Metrics
                loss_batch += loss.item()
                loss /= real_batch
                is_fst_loss = True
                loss.backward()
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
                optimizer.step()
                print(f"Train: {args.source}2{args.target}\t Epoch:{epoch:03d}  Batch:{batch_count:04d}\t Pred_Loss:{loss_batch/batch_count:.6f}")
                logger.add_scalar(tag="train/loss",scalar_value = loss_batch/batch_count,global_step = (epoch+1)*batch_count)

def val(epoch,model,best_metric,val_loader,logger,save_path,args):
    model.eval()
    loader_len = len(val_loader)
    loss = torch.tensor(0.,requires_grad=False).to(args.device)
    for cnt,batch in enumerate(val_loader): 
        #Get data
        V_obs = batch["V_observe"].to(args.device)
        A_obs = batch["A_observe"].to(args.device)
        V_gt  = batch["V_predict"].to(args.device)

        # forward
        V_obs =V_obs.permute(0,3,1,2)
        A_obs = A_obs.squeeze(0)
        V_pred,_,_ = model(V_obs,A_obs,None,None)
        V_pred = V_pred.permute(0,2,3,1)
        
        V_gt = V_gt.squeeze(0)
        V_pred = V_pred.squeeze(0)

        loss += bivariate_loss(V_pred,V_gt)

    loss /= loader_len
    #Metrics
    print(f"VALD : {args.source}2{args.target}\t ")
    
    logger.add_scalar(tag="val/loss",scalar_value = loss.item(),global_step = (epoch+1)*loader_len)

    if  loss.item() < best_metric['best_val_loss']:
        best_metric['best_val_loss'] =  loss.item()
        best_metric['best_val_epoch'] = epoch
        torch.save(model.state_dict(),str(save_path / 'val_best.pth'))  # OK
    print(f"VALD : Epoch:{epoch:03d}\tLoss:{loss.item():.6f}\tBest epoch:{best_metric['best_val_epoch']:03d}  Best Loss:{best_metric['best_val_loss']:.6f}")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    
    parser = STGCNN_Adaptive_parser()
    args = parser.parse_args()
    angles = []
    scales = []
    main(args)
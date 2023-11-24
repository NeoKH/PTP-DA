#!/usr/bin/env python3
# coding:utf8
"""
@file      log.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import sys
from visualdl import LogWriter
from torch.utils.tensorboard import SummaryWriter

# stdout and VisualDL logger
class StdoutLogger(object):
    def __init__(self, log):
        self.terminal = sys.stdout
        self.log = open(log, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

class VisualLogger(object):
    def __init__(self,logdir,mode = "tensorboard") -> None:
        self.mode = mode
        if self.mode in ["visualdl","both"]:
            self.visualdl = LogWriter(logdir=logdir)
        if self.mode in ["tensorboard","both"]:  
            self.tensorboard = SummaryWriter(log_dir=logdir)
            
    def add_scalar(self,tag,scalar_value,global_step):
        if self.mode in ["visualdl","both"]:
            self.visualdl.add_scalar(tag,step = global_step, value = scalar_value )
        if self.mode in ["tensorboard","both"]: 
            self.tensorboard.add_scalar(tag,global_step = global_step, scalar_value = scalar_value ) 
        
    def close(self):
        if self.mode in ["visualdl","both"]:
            self.visualdl.close()
        if self.mode in ["tensorboard","both"]:  
            self.tensorboard.close()


def record_on_batch(tag,epoch,batch,loss,pred_loss,align_loss):
    logstr = f"Train:{tag}\t Epoch:{epoch:03d}  Batch:{batch:03d}"
    logstr += f"\t Loss:{loss:.6f}"
    logstr += f"\t Pred_Loss:{pred_loss:.6f}"
    logstr += f"\t Align_Loss:{align_loss:.6f}"
    print(logstr)
    
def record_on_epoch(args,epoch,metrics_dict):
    logstr =  f"Train: {args.tag}\tEpoch:{epoch:03d}/{args.num_epochs}"
    logstr += f"\t Loss:{metrics_dict['total_loss'][-1]:.6f}"
    logstr += f"\t Pred_Loss:{metrics_dict['pred_loss'][-1]:.6f}"
    logstr += f"\t Align_Loss:{metrics_dict['align_loss'][-1]:.6f}"
    print(logstr)
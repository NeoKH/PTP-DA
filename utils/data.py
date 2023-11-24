#!/usr/bin/env python3
# coding:utf8
"""
@file      data.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import os
import math

import torch
import pickle
import numpy as np
from numpy import linalg
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.cluster import KMeans

def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len) # [0. 1. 2. 3. 4. 5. 6. 7.]
    import matplotlib.pyplot as plt
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def angles_cluster(name = "eth", method='kmeans',n_clusters=3,nums=10):
    # read csv
    df = pd.read_csv("./angles.csv",sep='\t')
    estimator = KMeans(n_clusters=n_clusters)
    
    data = df[df["name"]==name]
    data = data["rad"].to_numpy().reshape(-1, 1)
    row = []
    for num in range(nums):
        estimator.fit(data)
        centroids = estimator.cluster_centers_ 
        row.append(list(np.sort(centroids.squeeze())))
    thetas = np.mean(np.array(row),axis=0)
    return thetas

def read_eth_ucy(p_file):
    data = []
    with open(p_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def read_stcrowd(p_file):
    """
        return TIMESTAMP,TRACK_ID,X,Y
    """
    df = pd.read_csv(p_file)
    
    timestamps = np.sort(np.unique(df['TIMESTAMP'].values))
    ped_ids = np.sort(np.unique(df['TRACK_ID'].values))
    
    columns=['TIMESTAMP','TRACK_ID','X','Y']
    inters = []
    for ped_id in ped_ids:
        ped_data = df[df['TRACK_ID']==ped_id]
        times = np.sort(ped_data["TIMESTAMP"].values)
        if times[-1]-times[0]+1 == len(times):
            continue
        # TODO
        if np.max(times[1:]-times[:-1]) > 3:
            df.drop(index=df[df['TRACK_ID']==ped_id].index,inplace=True)
            continue
        for i in range(len(times)):
            if i==0:continue
            if times[i]-times[i-1]==2:
                pre = ped_data[ped_data["TIMESTAMP"]==times[i-1]]
                cur = ped_data[ped_data["TIMESTAMP"]==times[i]]
                mid_x = (pre.values[0][-2] + cur.values[0][-2]) / 2
                mid_y = (pre.values[0][-1] + cur.values[0][-1]) / 2
                inters.append([times[i-1]+1,ped_id,mid_x,mid_y])
        for i in range(len(times)):
            if i==0:continue
            if times[i]-times[i-1]==3:
                pre = ped_data[ped_data["TIMESTAMP"]==times[i-1]]
                cur = ped_data[ped_data["TIMESTAMP"]==times[i]]
                mid_x = (pre.values[0][-2] + cur.values[0][-2]) / 2
                mid_y = (pre.values[0][-1] + cur.values[0][-1]) / 2
                left_x = (pre.values[0][-2] + mid_x) / 2
                left_y = (pre.values[0][-1] + mid_y) / 2
                right_x = (cur.values[0][-2] + mid_x) / 2
                right_y = (cur.values[0][-1] + mid_y) / 2
                inters.append([times[i-1]+1,ped_id,left_x,left_y])
                inters.append([times[i-1]+2,ped_id,right_x,right_y])
    inters = pd.DataFrame(inters,columns=columns)
    df = pd.concat([df,inters],ignore_index=True)
    
    return df.values

def angle_statistic(
    src_root: str = "./data/origin",
    dst_root: str = "./data/statistic/angle",
    archive_name: str = "ETH_UCY_DA",
    data_name : str = "eth",
    data_type : str="test",
):
    src_path = Path(src_root) / archive_name / data_name / data_type
    assert src_path.exists()
    all_files = os.listdir(src_path)
    all_files = [os.path.join(src_path, _path) for _path in all_files]
    
    dst_path = Path(dst_root) / archive_name / data_type
    if not dst_path.exists():
        dst_path.mkdir(parents=True)
    dst_path = dst_path / f"{data_name}.pkl"
    
    angles_dict = {8:[],12:[],20:[],40:[]}
    for p_file in all_files:
        if archive_name.lower().find("eth")!=-1:
            data = read_eth_ucy(p_file)
        else:
            data = read_stcrowd(p_file)
        
        frames = np.unique(data[:,0]).tolist() # all frame timestamps
        # split data into sequence
        data_frames = []
        for frame in frames:
            data_frames.append(data[frame == data[:, 0], :])
        for obs_len in [8,12,20]: #
            num_sequences = int(math.ceil((len(frames) - obs_len + 1) / obs_len))
            temp_angles = []
            for idx in range(0,num_sequences * obs_len + 1, obs_len):
                '''处理单个Sequence'''
                if idx + obs_len >= num_sequences * obs_len + 1: break
                data_sequence = np.concatenate(data_frames[idx:idx + obs_len], axis=0)
                peds_in_seq = np.unique(data_sequence[:, 1])
                for ped_id in peds_in_seq:
                    '''处理单个行人'''
                    ped_seq = data_sequence[data_sequence[:, 1] ==ped_id, :]
                    # print(ped_seq.shape)
                    ped_seq = ped_seq.astype(np.float64)
                    ped_seq = np.around(ped_seq, decimals=4) # 保留小数点后4位
                    ped_seq_start = frames.index(ped_seq[0, 0]) - idx
                    ped_seq_end = frames.index(ped_seq[-1, 0]) - idx + 1
                    if ped_seq_end - ped_seq_start < obs_len:continue
                    coords = ped_seq[:, 2:]# (L,2)
                    dis = coords[1:,:] - coords[:-1,:] # dis存储两帧之间的x,y距离
                    dis_wo_still = np.array([x for x in dis if np.any(x!=0) ]) # 如果静止,x,y都为0,要排除这些点
                    if dis_wo_still.size==0: continue # 一直静止,则dis_wo_still为空,跳过
                    average_angle = np.mean(np.arctan2(dis_wo_still[:,1],dis_wo_still[:,0]))
                    temp_angles.append(average_angle)
            angles_dict[obs_len].extend(temp_angles)
        angles_dict[40].extend(angles_dict[8])
        angles_dict[40].extend(angles_dict[12])
        angles_dict[40].extend(angles_dict[20])

    with open(dst_path,"wb") as f:
        pickle.dump(angles_dict,f)

def get_sequence_angle(seq_rel):
    # 为了加快getitem的速度,这里直接使用seq_rel,即做完差的seq
    angles = []
    for ped_seq in seq_rel:
        dis = np.transpose(ped_seq)
        dis_wo_still = np.array([x for x in dis if np.any(x!=0) ]) # 如果静止,x,y都为0,要排除这些点
        if dis_wo_still.size==0: continue
        _angle = np.mean(np.arctan2(dis_wo_still[:,1],dis_wo_still[:,0])) # 一个人的平均角度
        angles.append(_angle)
    if len(angles)==0:
        return [0.]
    else:
        return angles

def get_sequence_average_angle(seq_rel):
    # 为了加快getitem的速度,这里直接使用相对seq,即做完差的seq
    # seq_rel: (N,2,7) N表示行人数,7表示len_observe-1
    angles = []
    for ped_seq in seq_rel:
        dis = np.transpose(ped_seq)
        dis_wo_still = np.array([x for x in dis if np.any(x!=0) ]) # 如果静止,x,y都为0,要排除这些点
        if dis_wo_still.size==0: continue
        _angle = np.mean(np.arctan2(dis_wo_still[:,1],dis_wo_still[:,0])) # 一个人的平均角度
        # 转换到第一象限
        # if _angle > np.pi/2 and _angle < np.pi:
        #     _angle = np.pi - _angle
        # elif _angle > -np.pi/2 and _angle < 0:
        #     _angle = - _angle
        # elif _angle > -np.pi and _angle < -np.pi/2:
        #     _angle = - _angle
        # else:
        #     _angle = _angle
        angles.append(_angle)
    if len(angles)==0:
        return 0.
    else:
        return np.mean(angles) # 所有人的角度转换到第一象限后的平均角度

def get_hist(file_path,bins = 45,angle_obs_len = 8):
    with open(file_path,'rb') as f:
        angle_dict = pickle.load(f)
    angles = np.array(angle_dict[angle_obs_len])
    hist,pillars  = np.histogram(angles,bins)
    angle_hist = {
        "prob": hist / np.sum(hist),
        "angle": (pillars[:-1] + pillars[1:]) /2
    }
    return angle_hist

def _get_hist(angles,bins=45):
    hist, pillars = np.histogram(angles, bins =bins)
    new_hist = []
    for x in hist:
        x = x if x!=0 else 0.01
        new_hist.append(x)
    hist = new_hist
    prob = torch.tensor(hist / np.sum(hist))
    # p = F.log_softmax(torch.tensor(p))
    angles = (pillars[:-1] + pillars[1:]) /2
    return prob,angles

def kde_compare(
    ps,pt,
    sname,tname,
    pimg="",dtype="angle",
    angle_obs_len=8,bins = 45
):
    with open(ps,'rb') as f:
        angle_dict = pickle.load(f)
    source_angles = np.array(angle_dict[angle_obs_len])
    with open(pt,'rb') as f:
        angle_dict = pickle.load(f)
    target_angles = np.array(angle_dict[angle_obs_len])

    p, _ = _get_hist(target_angles,bins)
    q, _ = _get_hist(source_angles,bins)
    kld = F.kl_div(p.log(),q,reduction='sum')

    p_angles = {
        'rad': target_angles,
        'name': [f"{tname}"]*len(target_angles),
    }
    p_df = pd.DataFrame(p_angles)
    q_angles = {
        'rad': source_angles,
        'name': [f"{sname}"]*len(source_angles),
    }
    q_df = pd.DataFrame(q_angles)
    _angles = pd.concat([q_df,p_df],ignore_index=True)
    sns.kdeplot(
        _angles,
        x='rad',
        hue = 'name',
        fill=True, 
        common_norm=False,
        # palette="crest",
        alpha=.5, linewidth=.5,
    )
    p_img = Path(pimg) / f"{dtype}_{sname}2{tname}.png"
    plt.title(f"KLD:{kld.item():.2f}")
    plt.savefig(p_img,format='png',transparent=True)
    plt.cla()
    
    print(f"{sname}2{tname} KLD : {kld.item():.2f}")
    return

def plot_5_datasets_distribution():
    # obj: scale or angle
    for i,obj in enumerate(["scale","angle"]):
        _p = f"./data/statistic/{obj}"
        names = ["eth","hotel","univ","zara1","zara2"]
        temp = []
        for name in names:
            with open(f"{_p}/ETH_UCY_DA/test/{name}.pkl","rb") as f:
                _dict = pickle.load(f)
            data = {
                'value': _dict[8],
                'dataset': [f"{name}"]*len(_dict[8]),
            }
            temp.append(pd.DataFrame(data))

        datas = pd.concat(temp,ignore_index=True)
        # plt.subplot(1,2,i+1)
        # 代码中的“...”代表省略的其他参数
        # ax = plt.subplot(111)
        # 设置刻度字体大小
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # 设置坐标标签字体大小
        # ax.set_xlabel(..., fontsize=20)
        # ax.set_ylabel(..., fontsize=20)
        # # 设置图例字体大小
        # ax.legend(..., fontsize=20)
        plt.rcParams.update({'font.size': 19})
        # plt.rcParams['mathtext.fontset']='stix'
        plt.rc('font',family='Times New Roman')
        sns.histplot(
            datas,
            x='value',
            hue = 'dataset',
            fill=True, 
            common_norm=False,
            kde=True,
            # palette="crest",
            stat="percent",
            alpha=.5, linewidth=.5,
        )
        plt.ylabel(f"percent (%)")
        if obj=="angle":
            plt.xlabel(f"angle (rad)")
        else:
            plt.xlabel(f"scale (m)")
        p_img = Path("./imgs") / f"{obj}_distribution.pdf"
        plt.savefig(p_img,format='pdf',transparent=True,bbox_inches='tight')
        plt.cla()



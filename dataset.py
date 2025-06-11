import os
import torch
import pandas as pd
import numpy as np
import pickle

from collections import Counter

from torch.utils import data
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.ndimage import gaussian_filter1d



def get_window_feature(data, sampling_interval):
    """ Get the windowed feature from the data.
    
    Args:
        data (numpy.ndarray): The data to be windowed.
        sampling_interval (int): The sampling interval for the window.
    
    Returns:
        numpy.ndarray: The windowed feature.
    """
    window_feature = []
    for ii in range(0, data.shape[0], sampling_interval):
        if ii + sampling_interval > data.shape[0]:
            break
        window_feature.append(data[ii:ii+sampling_interval])
    
    if ii == sampling_interval:
        window_feature.append(data[-sampling_interval:])
    
    window_feature = np.array(window_feature)
    
    
    return window_feature


def boundary_smooth(event_seq, smooth=None):

    event_seq = event_seq.T.squeeze()

    # boundary_seq = np.zeros_like(event_seq)

    boundary_seq = gaussian_filter1d(event_seq, smooth)
    
    # Normalize. This is ugly.
    temp_seq = np.zeros_like(boundary_seq)
    temp_seq[temp_seq.shape[0] // 2] = 1
    temp_seq[temp_seq.shape[0] // 2 - 1] = 1
    norm_z = gaussian_filter1d(temp_seq, smooth).max()
    boundary_seq[boundary_seq > norm_z] = norm_z
    boundary_seq /= boundary_seq.max()

    return boundary_seq



class SatellitePoL(data.Dataset):
    '''
    Satellite PoL dataset

    '''
    def __init__(self, data_dir, object_lists, labelfile, selected_features = None):
        self.data_dir = data_dir
        self.labelfile = labelfile

        self.seq_data = []
        self.seq_sup_labels = []
        self.seq_sup_boundary_labels = []
        self.seq_sub_labels = []
        self.seq_sub_boundary_labels = []
        self.seq_levels= []
        labeldata = pd.read_csv(self.labelfile)

        for obj_id in tqdm(object_lists):


            # 读文件路径
            data_path = os.path.join(self.data_dir, str(obj_id)+'.csv')
            data = pd.read_csv(data_path, index_col='TimeStep')

            # TODO
            # split sequence data into sub-sequences/frames

            # get the selected features
            data = data[selected_features].values
            # data = data[:120, :]

            # # get the windowed feature
            # data 

            # label data
            node_label = labeldata[(labeldata['ObjectID'] == obj_id) ]

            seq_sup_label = np.zeros((data.shape[0], 1), dtype=np.int16)
            seq_sub_label = np.zeros((data.shape[0], 1), dtype=np.int16)
            seq_level = np.zeros((data.shape[0], 1), dtype=np.int16)

            for ii in range(len(node_label)):
                cur_label = node_label.iloc[ii]['Label']
                cur_idx = node_label.iloc[ii]['TimeIndex']
                seq_sub_label[cur_idx:] = cur_label
                if ii == 1:
                    if (node_label.iloc[ii]['level']=='low'):
                        seq_level[0:] = 1
                    elif (node_label.iloc[ii]['level']=='mid'):
                        seq_level[0:] = 2
                    elif (node_label.iloc[ii]['level']=='high'):
                        seq_level[0:] = 3
                    else:
                        seq_level[0:] = 0

            seq_sub_boundary = np.zeros((data.shape[0], 1), dtype=np.float32)
            boundary_idx = np.where(seq_sub_label[1:] != seq_sub_label[:-1])[0]
            seq_sub_boundary[boundary_idx] = 1
            seq_sub_boundary[boundary_idx - 1] = 1

            if seq_sub_boundary.max() == 1:  
                t = boundary_smooth(seq_sub_boundary, 12*2)

            seq_sup_boundary = np.zeros((data.shape[0], 1), dtype=np.float32)
            boundary_idx = np.where(seq_sup_label[1:] != seq_sup_label[:-1])[0]
            seq_sup_boundary[boundary_idx] = 1
            seq_sup_boundary[boundary_idx - 1] = 1

            if seq_sup_boundary.max() == 1:  
                t = boundary_smooth(seq_sup_boundary, 12*2)

            # align length
            seq_sub_label = seq_sub_label[:data.shape[0]]
            seq_sup_label = seq_sup_label[:data.shape[0]]
            seq_sub_boundary = seq_sub_boundary[:data.shape[0]]
            seq_sup_boundary = seq_sup_boundary[:data.shape[0]]
            seq_level = seq_level[:data.shape[0]]


            # append feature and label
            self.seq_data.append(data)
            self.seq_sup_labels.append(seq_sup_label)
            self.seq_sub_labels.append(seq_sub_label)
            self.seq_levels.append(seq_level)


            self.seq_sup_boundary_labels.append(seq_sup_boundary)
            self.seq_sub_boundary_labels.append(seq_sub_boundary)

        self.seq_data = np.stack(self.seq_data).astype(np.float32)

        self.seq_sup_labels = np.stack(self.seq_sup_labels).astype(np.int64)
        self.seq_sub_labels = np.stack(self.seq_sub_labels).astype(np.int64)
        self.seq_sup_boundary_labels = np.stack(self.seq_sup_boundary_labels).astype(np.int64)
        self.seq_sub_boundary_labels = np.stack(self.seq_sub_boundary_labels).astype(np.int64)
        self.seq_levels = np.stack(self.seq_levels).astype(np.int64)

    def normalize(self, scaler=None):

        sample_num, seq_len, feature_num = self.seq_data.shape
        self.seq_data = self.seq_data.reshape(-1, feature_num)

        if scaler is None:
            scaler = StandardScaler()
            self.seq_data = scaler.fit_transform(self.seq_data)
        else:
            self.seq_data = scaler.transform(self.seq_data)

        self.seq_data = self.seq_data.reshape(sample_num, seq_len, feature_num)

        return scaler

    def count_sup_label(self):
        return Counter(list(self.seq_sup_labels.flatten()))
    

    def count_sub_label(self):
        return Counter(list(self.seq_sub_labels.flatten()))
    
    def __len__(self):
        return len(self.seq_data)
    
    def __getitem__(self, idx):

        X = self.seq_data[idx]
        target = {'sup_label': self.seq_sup_labels[idx],
                  'sub_label': self.seq_sub_labels[idx],
                  'sup_boundary_label': self.seq_sup_boundary_labels[idx],
                  'sub_boundary_label': self.seq_sub_boundary_labels[idx],
                  'level': self.seq_levels[idx]
                  }

        return X, target


    

if __name__ == "__main__":
    
    ## test satellite dataset

    # data_dir = "sate_data/train/"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(
        current_dir,  # 当前脚本所在目录
        '..',  # 返回上一级目录
        'ToHillstate',  # 进入同级的 "Create Mega Satellite" 文件夹
        'Result',  # 进入 "InPutData" 文件夹
        'div',

    )
    labelfile = "sate_data/300_train_labels.csv"
    # 离心率，半长轴，轨道倾角，升交点赤经，近地点幅角，真近点角，维度，经度，高度
    selected_features = ["x", "y", "z", "Vx", "Vy", "Vz"]

    labeldata = pd.read_csv(labelfile)
    object_ids = labeldata['ObjectID'].unique()
    train_ids, test_ids = train_test_split(object_ids, 
                                            test_size=0.2, 
                                            random_state=42)
    
    train_dataset = SatellitePoL(data_dir, train_ids, labelfile, selected_features)
    scaler = train_dataset.normalize()
    test_dataset = SatellitePoL(data_dir, test_ids, labelfile, selected_features)
    test_dataset.normalize(scaler)

    train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    for X, target in train_dataloader:
        print(X.shape, target.shape)
        break

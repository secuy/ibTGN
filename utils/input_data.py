import os
from scipy.special import expit
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from dipy.viz import actor, window

def load_fiber_data(human_root_path, primate_root_path, human_sub_num, human_start_num, primate_sub_num, primate_start_num):
    adjs = []
    features = []
    # human
    human_real_num = 0
    primate_real_num = 0
    for idx in range(human_start_num, human_start_num + human_sub_num):
        cur_mat_path = human_root_path+"sub-{}_dist_mat.npy".format(str(idx).zfill(3))
        cur_feat_path = human_root_path+"sub-{}_mean_fibers.npy".format(str(idx).zfill(3))
        if not os.path.exists(cur_mat_path):
            print("sub-" + str(idx) + " not exist")
            continue
        else:
            print("read:human_sub-" + str(idx))
            human_real_num += 1
        dist_mat = np.load(cur_mat_path)
        mean_fibers = np.load(cur_feat_path)
        mean_fibers = mean_fibers.reshape(mean_fibers.shape[0], mean_fibers.shape[1] * mean_fibers.shape[2])
        scaler = StandardScaler()
        mean_fibers = scaler.fit_transform(mean_fibers)
        # Step 1: 处理距离矩阵
        adjacency_matrix = np.where(dist_mat <= 1.3, 1, 0)
        # adjacency_matrix = dist_mat
        for idx, line in enumerate(adjacency_matrix):
            if line[idx] != 1:
                line[idx] = 1
        tot = 0
        for idx, line in enumerate(adjacency_matrix):
            tot += np.sum(line != 0)
        # Step 2: 处理特征矩阵
        feature = mean_fibers

        adjs.append(adjacency_matrix)
        features.append(feature)
    for idx in range(primate_start_num, primate_start_num + primate_sub_num):
        cur_mat_path = primate_root_path + "sub-{}_dist_mat.npy".format(str(idx).zfill(3))
        cur_feat_path = primate_root_path + "sub-{}_mean_fibers.npy".format(str(idx).zfill(3))
        if not os.path.exists(cur_mat_path):
            print("sub-" + str(idx) + " not exist")
            continue
        else:
            print("read:primate_sub-" + str(idx))
            primate_real_num += 1
        dist_mat = np.load(cur_mat_path)
        mean_fibers = np.load(cur_feat_path)
        mean_fibers = mean_fibers.reshape(mean_fibers.shape[0], mean_fibers.shape[1] * mean_fibers.shape[2])
        scaler = StandardScaler()
        mean_fibers = scaler.fit_transform(mean_fibers)
        # Step 1: 处理距离矩阵
        adjacency_matrix = np.where(dist_mat <= 0.65, 1, 0)
        # adjacency_matrix = dist_mat
        for idx, line in enumerate(adjacency_matrix):
            if line[idx] != 1:
                line[idx] = 1
        tot = 0
        for idx, line in enumerate(adjacency_matrix):
            tot += np.sum(line != 0)
        # Step 2: 处理特征矩阵
        feature = mean_fibers

        adjs.append(adjacency_matrix)
        features.append(feature)
    print("human sub real number:{}".format(human_real_num))
    print("primate sub real number:{}".format(primate_real_num))
    adjs = np.array(adjs)
    features = np.array(features)
    return adjs, features
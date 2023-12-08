# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:15:59 2023

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from sklearn.cluster import DBSCAN

plt.close('all')

def clus_agn(ind):
    rd_root = rd_data[ind]
    dism = np.linalg.norm(rd_data[ind]-rd_data, axis=1)
    disleaf = dism < eps
    neighbour = idk[disleaf].tolist()
    if (len(neighbour) > (minNumSamp-1)):
    
        for j in range(rd_data.shape[0]):
            rd_leaf = rd_data[j]
            rd_dif = rd_leaf - rd_root
            dis = norm(rd_dif, 2)
            if (dis == 0):
                continue
            if (dis < eps):
                if (cluster_ind[j]>cluster_ind[i]):
                    cluster_ind[j]=cluster_ind[i]
                    # if (j<i):
                    #     clus_agn(j)
                else:
                    cluster_ind[i]=cluster_ind[j]
                # clus_agn(i)

rd_data = np.array([[-3, 2], [1, 1], [-1, 4], [-1, -2], [3, 4], [-4, 2], [-3, -4], [1, 2], [-3, 4]], dtype='float64')

rd_data = np.array([[1, 2], [2, 2], [2, 4], [2, -4], [4, 4], [4, -2], [6, 1], [6, 2], [8, 4]], dtype='float64')



# rd_data = np.load("rd_data57_noise.npy")
# rd_data = rd_data[:, 0:2]

# np.random.shuffle(rd_data)

for i in range(rd_data.shape[0]):
    if (rd_data[i, 1] > 127):
        rd_data[i, 1] -=256
cluster_ind = np.zeros(rd_data.shape[0], dtype='int64')

eps = 1.2
minNumSamp = 4

rangeRes = 0.075
velRes = 0.174075311942959

# rd_data[:, 0] = rd_data[:, 0]*rangeRes
# rd_data[:, 1] = rd_data[:, 1]*velRes

# rd_data = np.load("tset2.npy")




# pc_file = np.loadtxt("PC_0000000034.txt")
# rd_data = np.zeros((len(pc_file[pc_file[:, 0] == 1]), 2), dtype = 'float32')
# rd_data[:, 0]= pc_file[pc_file[:, 0] == 1][:, 7]
# rd_data[:, 1]= pc_file[pc_file[:, 0] == 1][:, 11]
# gt_cluIdx = pc_file[pc_file[:, 0] == 1][:, 15]
# cluster_ind = np.zeros(rd_data.shape[0], dtype='int64')


dbscan_clu = DBSCAN(eps, min_samples=minNumSamp, metric='euclidean').fit(rd_data)
idk = np.arange(rd_data.shape[0])


for i in range(rd_data.shape[0]):
    rd_root = rd_data[i]
    
    dism = np.linalg.norm(rd_data[i]-rd_data, axis=1)
    disleaf = dism < eps
    neighbour = idk[disleaf].tolist()
    
    # if (len(neighbour) < minNumSamp):
    #     cluster_ind[i] = -1
    #     continue
    
    cntrl = 0
    if (i==314):
        aa = 1
    
    # if (i==10):
    #     bb = 1
    # j=0
    # while (j < rd_data.shape[0]):
    for j in range(rd_data.shape[0]):
        
        if (i==166 & j==26):
            bb=1
        rd_leaf = rd_data[j]
        rd_dif = rd_leaf - rd_root
        dis = norm(rd_dif, 2)
        if (dis == 0):
            continue
        if (j==26 & i==27):
            bb=1
        if (dis < eps):
            if (cluster_ind[j]>cluster_ind[i]):
                cluster_ind[j]=cluster_ind[i]
                clus_agn(j)
                
                
                if (j == 7):
                    aa = 1
                
                    
                    # rd = rd_data[j]
                    # for k in range(rd_data.shape[0]):
                    #     rd_leaf1 = rd_data[k]
                    #     rd_dif1 = rd_leaf1 - rd
                    #     dis1 = norm(rd_dif1, 2)
                    #     if (dis1 < eps):
                    #         if (cluster_ind[k]>cluster_ind[j]):
                    #             cluster_ind[k]=cluster_ind[j]
                    #         else:
                    #             cluster_ind[j]=cluster_ind[k]
            else:
                cluster_ind[i]=cluster_ind[j]
                clus_agn(i)
                # dism = np.linalg.norm(rd_data[i]-rd_data, axis=1)
                # disleaf = dism < eps
                # neighbour = idk[disleaf].tolist()
                
                # if (len(neighbour) > minNumSamp):
                #     clus_agn(i)
                
                if (i == 7):
                    aa = 1
                # rd = cluster_ind[j]
                # for k in range(rd_data.shape[0]):
                #     rd_leaf1 = rd_data[k]
                #     rd_dif1 = rd_leaf1 - rd
                #     dis1 = norm(rd_dif1, 2)
                #     if (dis1 < eps):
                #         if (cluster_ind[k]>cluster_ind[i]):
                #             cluster_ind[k]=cluster_ind[i]
                #         else:
                #             cluster_ind[i]=cluster_ind[k]
                #     else:
                #         cluster_ind[i]=cluster_ind[j]
                # j=-1
            
            
                
        else:
            if ((cluster_ind[i] == cluster_ind[j]) & (j>i)):
                cluster_ind[j]+=1
                if (i != 0 ):
                    if (cluster_ind [j] < np.amax(cluster_ind[0:j])):
                        cluster_ind[j] = np.amax(cluster_ind[0:j])+1
                # j=0
                # if (i != 0):
                #     if (cluster_ind[j]>(np.amax(cluster_ind[0:i])+1)):
                #         cluster_ind[j] = (np.amax(cluster_ind[0:i])+1)
        # j += 1         
    
num_clus = np.unique(cluster_ind)
clu_ind = 0

for clu in (np.unique(cluster_ind)):
    if (np.count_nonzero(cluster_ind == clu) < minNumSamp):
        cluster_ind[cluster_ind == clu] = -1
        continue
        
    if (clu != clu_ind):
        cluster_ind[cluster_ind == clu] = clu_ind
    clu_ind += 1
        
    

for clu in (np.unique(cluster_ind)):
    clus = rd_data[np.where(cluster_ind == clu)]
    plt.figure(1, dpi=200)
    plt.title("my_algo")
    plt.scatter(clus[:,0], clus[:,1])

plt.plot(rd_data[118:120, 0], rd_data[118:120, 1], 'r*')
plt.show()
plt.grid(True)


for clu in (np.unique(cluster_ind)):
    clus = rd_data[np.where(dbscan_clu.labels_ == clu)]
    plt.figure(2, dpi=200)
    plt.title("DBSCAN")
    plt.scatter(clus[:,0], clus[:,1])

plt.plot(rd_data[118:120, 0], rd_data[118:120, 1], 'r*')
plt.show()
plt.grid(True)

# dbscan_clu = DBSCAN(eps, min_samples=minNumSamp, metric='euclidean').fit(rd_data)


import numpy as np
import torch
import os
import csv

def common_loss2(adj_1, adj_2):
    adj_1 = adj_1 - torch.eye(adj_1.shape[0]).cuda()
    adj_2 = torch.eye(adj_2.shape[0]).cuda() - adj_2
    cost = torch.sum((adj_1 - adj_2) ** 2)
    cost = torch.exp(-cost)
    return cost


def common_loss(adj_1, adj_2):
    adj_1 = adj_1 * (1 - torch.eye(adj_1.shape[0]).cuda())
    adj_2 = torch.eye(adj_2.shape[0]).cuda() - adj_2
    cost = torch.sum((adj_1 - adj_2) ** 2)
    cost = torch.exp(-cost)
    return cost

def dependence_loss(adj_1, adj_2):
    node_num = adj_1.shape[0]
    R = torch.eye(node_num) - (1/node_num) * torch.ones(node_num, node_num)
    adj_1 = adj_1 * (1 - torch.eye(adj_1.shape[0]).cuda())
    adj_2 = torch.eye(adj_2.shape[0]).cuda() - adj_2
    K1 = torch.mm(adj_1, adj_1.T)
    K2 = torch.mm(adj_2, adj_2.T)
    RK1 = torch.mm(R.cuda(), K1)
    RK2 = torch.mm(R.cuda(), K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def loss_dependence2(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def load_spatialmatrix(dataset, num_node):
    files = {'PEMSD3': ['PeMSD3/pems03.npz', 'PeMSD3/distance.csv', 'pems03'],
             'PEMSD4': ['PeMSD4/pems04.npz', 'PeMSD4/distance.csv', 'pems04'],
             'PEMSD7': ['PeMSD7/pems07.npz', 'PeMSD7/distance.csv', 'pems07'],
             'PEMSD8': ['PeMSD8/pems08.npz', 'PeMSD8/distance.csv', 'pems08'],}
    filename = dataset
    file = files[filename]
    filepath = "../data/"

    if not os.path.exists(f'../data/{file[2]}_spatial_distance.npy'):
        with open(filepath + file[1], 'r') as fp:
            dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
            file_csv = csv.reader(fp)
            for line in file_csv:
                break
            for line in file_csv:
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
            np.save(f'../data/{file[2]}_spatial_distance.npy', dist_matrix)

    dist_matrix = np.load(f'../data/{file[2]}_spatial_distance.npy')
    std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = 10  # sp 10 dtw 0.1
    sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
    sp_matrix[sp_matrix <= 0] = 0.
    sp_matrix[sp_matrix > 0] = 1.
    sp_matrix = torch.from_numpy(sp_matrix).to(torch.float32)
    ds = torch.sum(sp_matrix, dim=0)
    Ds = torch.diag(torch.rsqrt(ds))
    sp_matrix = torch.matmul(Ds, torch.matmul(sp_matrix, Ds))

    return sp_matrix

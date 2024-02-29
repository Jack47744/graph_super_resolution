import torch
import torch.nn as nn
import numpy as np

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def get_device():
    # Check for CUDA GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Check for Apple MPS (requires PyTorch 1.12 or later)
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    # Fallback to CPU
    else:
        return torch.device("cpu")

device = get_device()

class GraphUnpool(nn.Module):

    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]]).to(device)
        new_X[idx] = X
        return A, new_X

    
class GraphPool(nn.Module):

    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        # scores = torch.abs(scores)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores/100)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k*num_nodes))
        new_X = X[idx, :]
        values = torch.unsqueeze(values, -1)
        new_X = torch.mul(new_X, values)
        A = A[idx, :]
        A = A[:, idx]
        return A, new_X, idx


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=0)

    def forward(self, A, X):
        # print(X.device)
        # print(self.proj.weight.device)
        X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        return X

class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim=300):
        super(GraphUnet, self).__init__()
        self.ks = ks
       
        self.start_gcn = GCN(in_dim, dim)
        self.bottom_gcn = GCN(dim, dim)
        self.end_gcn = GCN(2*dim, out_dim)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)

        self.down_gcns = nn.ModuleList([GCN(dim, dim) for i in range(self.l_n)])
        self.up_gcns = nn.ModuleList([GCN(dim, dim) for i in range(self.l_n)])
        self.pools = nn.ModuleList([GraphPool(ks[i], dim) for i in range(self.l_n)])
        self.unpools = nn.ModuleList([GraphUnpool() for i in range(self.l_n)])

        # for i in range(self.l_n):
        #     self.down_gcns.append(GCN(dim, dim))
        #     self.up_gcns.append(GCN(dim, dim))
        #     self.pools.append(GraphPool(ks[i], dim))
        #     self.unpools.append(GraphUnpool())

    def forward(self, A, X):
        # print('start_gcn device: ', self.start_gcn.device)
        adj_ms = []
        indices_list = []
        down_outs = []
        X = self.start_gcn(A, X)
        start_gcn_outs = X
        org_X = X
        for i in range(self.l_n):
            X = self.down_gcns[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
            # print('Down_gcns, i = ', i)
        
        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)
        
        return X, start_gcn_outs[:, :268]
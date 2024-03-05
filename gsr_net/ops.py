import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from layers import GCNLayer

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
    

class MultiHeadGAT(nn.Module):
    def __init__(self, in_features, out_features, heads=4, activation=None, residual=False, layer_norm=False):
        super(MultiHeadGAT, self).__init__()
        self.in_features = in_features

        # print(f"Multihead: {heads}")

        self.out_features = out_features // heads  # Adjust the size per head
        assert out_features % heads == 0
        self.heads = heads
        self.activation = activation
        self.residual = residual
        self.layer_norm = layer_norm

        # Initialize parameters for each head
        self.weights = nn.Parameter(torch.FloatTensor(heads, in_features, self.out_features))
        self.biases = nn.Parameter(torch.FloatTensor(heads, self.out_features))
        self.phis = nn.Parameter(torch.FloatTensor(heads, 2 * self.out_features, 1))

        if self.layer_norm:
            self.norm = nn.LayerNorm(out_features)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.heads):
            nn.init.xavier_uniform_(self.weights[i])
            nn.init.zeros_(self.biases[i])
            nn.init.xavier_uniform_(self.phis[i])

    def forward(self, adj, input):
        head_outputs = []
        N = adj.size(0)
        print(f"N = {N}")
        for i in range(self.heads):
            x_prime = torch.matmul(input, self.weights[i]) + self.biases[i]

            a_input = torch.cat([x_prime.repeat(1, N).view(N * N, -1), x_prime.repeat(N, 1)], dim=1)
            S = torch.matmul(a_input, self.phis[i]).view(N, N)
            S = F.leaky_relu(S, negative_slope=0.2)
            # S = F.gelu(S)

            mask = (adj + torch.eye(N, device=adj.device)) > 0
            S_masked = torch.where(mask, S, torch.full_like(S, -1e9))
            attention = F.softmax(S_masked, dim=1)
            h = torch.matmul(attention, x_prime)

            head_outputs.append(h)

            # print(h.shape)
        h_concat = torch.cat(head_outputs, dim=-1)
        # print(h_concat.shape)

        if self.residual:
            h_concat += input

        if self.activation:
            h_concat = self.activation(h_concat)

        if self.layer_norm:
            h_concat = self.norm(h_concat)

        return h_concat
    
class GAT(nn.Module):
    """
    A basic implementation of the GAT layer.

    This layer applies an attention mechanism in the graph convolution process,
    allowing the model to focus on different parts of the neighborhood
    of each node.
    """
    def __init__(self, in_features, out_features, activation = None, residual=False, layer_norm=True):
        super(GAT, self).__init__()
        # Initialize the weights, bias, and attention parameters as
        # trainable parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.phi = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.activation = activation
        self.residual = residual
        self.layer_norm = layer_norm
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / np.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

        # stdv = 1. / np.sqrt(self.phi.size(1))
        # self.phi.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, adj, input):
        x_prime = input @ self.weight  + self.bias

        N = adj.size(0)
        a_input = torch.cat([x_prime.repeat(1, N).view(N * N, -1), x_prime.repeat(N, 1)], dim=1)
        S = (a_input @ self.phi).view(N, N)
        S = F.leaky_relu(S, negative_slope=0.2)

        mask = (adj + torch.eye(adj.size(0))) > 0
        S_masked = torch.where(mask, S, torch.full_like(S, -1e9))
        attention = F.softmax(S_masked, dim=1)
        h = attention @ x_prime

        if self.residual:
            h = input + h

        if self.activation:
            h = self.activation(h)
        return h


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim=300):
        super(GraphUnet, self).__init__()
        self.ks = ks
       
        self.start_gcn = MultiHeadGAT(in_dim, dim)
        # self.bottom_gcn = GAT(dim, dim)
        self.bottom_gcn = MultiHeadGAT(dim, dim, residual=True)
        self.end_gcn = MultiHeadGAT(2*dim, out_dim)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)

        # self.down_gcns = nn.ModuleList([GAT(dim, dim) for i in range(self.l_n)])
        # self.up_gcns = nn.ModuleList([GAT(dim, dim) for i in range(self.l_n)])
        self.down_gcns = nn.ModuleList([MultiHeadGAT(dim, dim, residual=True) for i in range(self.l_n)])
        self.up_gcns = nn.ModuleList([MultiHeadGAT(dim, dim, residual=True) for i in range(self.l_n)])
        self.pools = nn.ModuleList([GraphPool(ks[i], dim) for i in range(self.l_n)])
        self.unpools = nn.ModuleList([GraphUnpool() for i in range(self.l_n)])
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=True) for _ in range(self.l_n)])

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
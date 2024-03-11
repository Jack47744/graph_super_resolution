import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from  utils import get_device

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
        X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        return X
    
class GAT(nn.Module):
    """
    A basic implementation of the GAT layer.

    This layer applies an attention mechanism in the graph convolution process,
    allowing the model to focus on different parts of the neighborhood
    of each node.

    Attributes:
    weight (Tensor): The weight matrix of the layer.
    bias (Tensor): The bias vector of the layer.
    phi (Tensor): The attention parameter of the layer.
    activation (function): The activation function to be used.
    residual (bool): Whether to use residual connections.
    out_features (int): The number of output features of the layer.
    """
    def __init__(self, in_features, out_features, activation = None, residual = False):
        super(GAT, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.phi = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.activation = activation
        self.reset_parameters()
        self.residual = residual
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        stdv = 1. / np.sqrt(self.phi.size(1))
        self.phi.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        """
        Forward pass of the GAT layer.

        Parameters:
        input (Tensor): The input features of the nodes.
        adj (Tensor): The adjacency matrix of the graph.

        Returns:
        Tensor: The output features of the nodes after applying the GAT layer.
        """
        x_prime = input @ self.weight  + self.bias

        N = adj.size(0)
        a_input = torch.cat([x_prime.repeat(1, N).view(N * N, -1), x_prime.repeat(N, 1)], dim=1)
        S = (a_input @ self.phi).view(N, N)
        S = F.leaky_relu(S, negative_slope=0.2)

        mask = (adj + torch.eye(adj.size(0), device = device)) > 0
        S_masked = torch.where(mask, S, torch.full_like(S, -1e9))
        attention = F.softmax(S_masked, dim=1)
        h = attention @ x_prime

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = input + h

        return h


class GraphUnet(nn.Module):
    """
    Our implementation of the Graph Unet model

    Attributes:
    ks (list): The list of pooling sizes.
    in_dim (int): The number of input features.
    out_dim (int): The number of output features.
    dim (int): The number of features in the hidden layers.
    start_gcn (GCN): The first GCN layer.
    bottom_gcn (GCN): The bottom GCN layer.
    end_gcn (GCN): The last GCN layer.
    down_gcns (list): The list of GCN layers in the downsampling path.
    up_gcns (list): The list of GCN layers in the upsampling path.
    pools (list): The list of pooling layers.
    unpools (list): The list of unpooling layers.
    l_n (int): The number of pooling layers.
    """

    def __init__(self, ks, in_dim, out_dim, dim=300):
        super(GraphUnet, self).__init__()
        self.ks = ks
        dim = out_dim

        self.start_gcn = GAT(in_dim, dim, activation=F.leaky_relu)
        self.bottom_gcn = GAT(dim, dim, residual=True, activation=F.leaky_relu)
        self.end_gcn = GAT(2*dim, out_dim, activation=F.leaky_relu)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)

        self.down_gcns = nn.ModuleList([GAT(dim, dim, residual=True, activation=F.leaky_relu) for i in range(self.l_n)])
        self.up_gcns = nn.ModuleList([GAT(dim, dim, residual=True, activation=F.leaky_relu) for i in range(self.l_n)])
        self.pools = nn.ModuleList([GraphPool(ks[i], dim) for i in range(self.l_n)])
        self.unpools = nn.ModuleList([GraphUnpool() for i in range(self.l_n)])

    def forward(self, A, X):
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


            # Start Before Edit
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
            # End Before Edit

        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)
        
        return X, start_gcn_outs[:, :268]
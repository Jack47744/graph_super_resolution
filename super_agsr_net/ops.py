import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
<<<<<<< HEAD:gsr_net/ops.py
from layers import GCNLayer

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def get_device():
    # Check for CUDA GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Check for Apple MPS (requires PyTorch 1.12 or later)
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    # Fallback to CPU
    else:
        return torch.device("cpu")
=======
from  utils import get_device
>>>>>>> 143fbb4bca6ad02b743607de1b8f0d5ffb7f3134:super_agsr_net/ops.py

device = get_device()

class GraphUnpool(nn.Module):

    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        # print(f"GraphUnpool X: {X.shape}")
        # print(f"GraphUnpool A: {A.shape}")
        # print(f"GraphUnpool idx: {idx.shape}")
        new_X = torch.zeros([A.shape[0], A.shape[1], X.shape[2]]).to(device)
        # print(f"GraphUnpool new_X: {new_X.shape}")
        new_X = new_X.scatter_(1, idx.unsqueeze(2).expand(-1, -1, X.shape[2]), X)
        # new_X[idx] = X
        # print(f"GraphUnpool new_X: {new_X.shape}")
        return A, new_X

class GraphPool(nn.Module):

    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
<<<<<<< HEAD:gsr_net/ops.py
        # scores = torch.abs(scores)
        # print(f"GraphPool scores: {scores.shape}")
        scores = torch.squeeze(scores, dim=2)
        # print(f"GraphPool scores: {scores.shape}")
=======
        scores = torch.squeeze(scores)
>>>>>>> 143fbb4bca6ad02b743607de1b8f0d5ffb7f3134:super_agsr_net/ops.py
        scores = self.sigmoid(scores/100)
        num_nodes = A.shape[1]
        values, idx = torch.topk(scores, int(self.k*num_nodes))
        batch_indices = np.arange(X.shape[0])[:, None]
        new_X = X[batch_indices, idx, :]
        values = torch.unsqueeze(values, -1)
        new_X = torch.mul(new_X, values)
        A = A[batch_indices, idx, :]
        A = A[batch_indices, :, idx]
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

        batch_size, N, _ = adj.size()

        for i in range(self.heads):
            # print(f"MultiHeadGAT input: {input.shape}")
            # print(f"MultiHeadGAT weights: {self.weights[i].shape}")
            # print(f"MultiHeadGAT biases: {self.biases[i].shape}")
            x_prime = torch.matmul(input, self.weights[i]) + self.biases[i]

            # print(f"MultiHeadGAT x_prime: {x_prime.shape}")
            a_input = torch.cat([x_prime.repeat(1, 1, N).view(batch_size, N * N, -1), x_prime.repeat(1, N, 1)], dim=2)
            # print(f"phi: {self.phis[i].shape}")
            S = torch.matmul(a_input, self.phis[i]).view(batch_size, N, N)
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

    Attributes:
    weight (Tensor): The weight matrix of the layer.
    bias (Tensor): The bias vector of the layer.
    phi (Tensor): The attention parameter of the layer.
    activation (function): The activation function to be used.
    residual (bool): Whether to use residual connections.
    out_features (int): The number of output features of the layer.
    """
<<<<<<< HEAD:gsr_net/ops.py
    def __init__(self, in_features, out_features, activation = None, residual=False, layer_norm=True):
=======
    def __init__(self, in_features, out_features, activation = None, residual = False):
>>>>>>> 143fbb4bca6ad02b743607de1b8f0d5ffb7f3134:super_agsr_net/ops.py
        super(GAT, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.phi = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.activation = activation
        self.residual = residual
        self.layer_norm = layer_norm
        self.reset_parameters()
        self.residual = residual
        self.out_features = out_features
        
    def reset_parameters(self):
        # stdv = 1. / np.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

        # stdv = 1. / np.sqrt(self.phi.size(1))
        # self.phi.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, adj, input):
<<<<<<< HEAD:gsr_net/ops.py
=======
        """
        Forward pass of the GAT layer.

        Parameters:
        input (Tensor): The input features of the nodes.
        adj (Tensor): The adjacency matrix of the graph.

        Returns:
        Tensor: The output features of the nodes after applying the GAT layer.
        """
>>>>>>> 143fbb4bca6ad02b743607de1b8f0d5ffb7f3134:super_agsr_net/ops.py
        x_prime = input @ self.weight  + self.bias

        N = adj.size(0)
        a_input = torch.cat([x_prime.repeat(1, N).view(N * N, -1), x_prime.repeat(N, 1)], dim=1)
        S = (a_input @ self.phi).view(N, N)
        S = F.leaky_relu(S, negative_slope=0.2)

        mask = (adj + torch.eye(adj.size(0), device = device)) > 0
        S_masked = torch.where(mask, S, torch.full_like(S, -1e9))
        attention = F.softmax(S_masked, dim=1)
        h = attention @ x_prime

<<<<<<< HEAD:gsr_net/ops.py
        if self.residual:
            h = input + h

        if self.activation:
            h = self.activation(h)
=======
        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = input + h

>>>>>>> 143fbb4bca6ad02b743607de1b8f0d5ffb7f3134:super_agsr_net/ops.py
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
<<<<<<< HEAD:gsr_net/ops.py
       
        self.start_gcn = MultiHeadGAT(in_dim, dim)
        # self.bottom_gcn = GAT(dim, dim)
        self.bottom_gcn = MultiHeadGAT(dim, dim, residual=True)
        self.end_gcn = MultiHeadGAT(2*dim, out_dim)
=======
        dim = out_dim

        self.start_gcn = GAT(in_dim, dim, activation=F.leaky_relu)
        self.bottom_gcn = GAT(dim, dim, residual=True, activation=F.leaky_relu)
        self.end_gcn = GAT(2*dim, out_dim, activation=F.leaky_relu)
>>>>>>> 143fbb4bca6ad02b743607de1b8f0d5ffb7f3134:super_agsr_net/ops.py
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)

<<<<<<< HEAD:gsr_net/ops.py
        # self.down_gcns = nn.ModuleList([GAT(dim, dim) for i in range(self.l_n)])
        # self.up_gcns = nn.ModuleList([GAT(dim, dim) for i in range(self.l_n)])
        self.down_gcns = nn.ModuleList([MultiHeadGAT(dim, dim, residual=True) for i in range(self.l_n)])
        self.up_gcns = nn.ModuleList([MultiHeadGAT(dim, dim, residual=True) for i in range(self.l_n)])
=======
        self.down_gcns = nn.ModuleList([GAT(dim, dim, residual=True, activation=F.leaky_relu) for i in range(self.l_n)])
        self.up_gcns = nn.ModuleList([GAT(dim, dim, residual=True, activation=F.leaky_relu) for i in range(self.l_n)])
>>>>>>> 143fbb4bca6ad02b743607de1b8f0d5ffb7f3134:super_agsr_net/ops.py
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
            # print(f"pool A: {A.shape}")
            # print(f"pool X: {X.shape}")
            # print(f"pool idx: {idx.shape}")
            indices_list.append(idx)
        
        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A, idx = adj_ms[up_idx], indices_list[up_idx]
            # print(f"unpool A: {A.shape}")
            # print(f"unpool X: {X.shape}")
            # print(f"unpool idx: {idx.shape}")
            A, X = self.unpools[i](A, X, idx)


            # Start Before Edit
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
<<<<<<< HEAD:gsr_net/ops.py
        X = torch.cat([X, org_X], dim=2)
        # print("moving to end gcn")
        # print(f"end_gcn X: {X.shape}")
        # print(f"end_gcn A: {A.shape}")
=======
            # End Before Edit

        X = torch.cat([X, org_X], 1)
>>>>>>> 143fbb4bca6ad02b743607de1b8f0d5ffb7f3134:super_agsr_net/ops.py
        X = self.end_gcn(A, X)
        
        return X, start_gcn_outs[:, :, :268]
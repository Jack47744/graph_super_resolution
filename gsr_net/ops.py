import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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
        self.scale = nn.Parameter(torch.tensor([0.01]))
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        # scores = torch.abs(scores)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores * self.scale)
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
    
class GAT(nn.Module):
    """
    A basic implementation of the GAT layer.

    This layer applies an attention mechanism in the graph convolution process,
    allowing the model to focus on different parts of the neighborhood
    of each node.
    """
    def __init__(self, in_features, out_features, activation = None, residual = False):
        super(GAT, self).__init__()
        # Initialize the weights, bias, and attention parameters as
        # trainable parameters
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
        ############# Your code here ############
        ## 1. Apply linear transformation and add bias
        ## 2. Compute the attention scores utilizing the previously
        ## established mechanism.
        ## Note: Keep in mind that employing matrix notation can
        ## optimize this process.
        ## 3. Compute mask based on adjacency matrix
        ## 4. Apply mask to the pre-attention matrix
        ## 5. Compute attention weights using softmax
        ## 6. Aggregate features based on attention weights
        ## Note: name the last line as `h`
        ## (9-10 lines of code)
        x_prime = input @ self.weight  + self.bias

        N = adj.size(0)
        a_input = torch.cat([x_prime.repeat(1, N).view(N * N, -1), x_prime.repeat(N, 1)], dim=1)
        S = (a_input @ self.phi).view(N, N)
        S = F.leaky_relu(S, negative_slope=0.2)

        mask = (adj + torch.eye(adj.size(0), device = device)) > 0
        S_masked = torch.where(mask, S, torch.full_like(S, -1e9))
        attention = F.softmax(S_masked, dim=1)
        # print(attention.shape, x_prime.shape)
        h = attention @ x_prime

        if self.activation:
            h = self.activation(h)

        if self.residual:
            # print(input.shape, h.shape)
            h = input + h

        #########################################
        return h


class GraphUnet(nn.Module):

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
        # self.up_gcns = nn.ModuleList([GAT(2*dim, dim, residual=True, activation=F.relu) for i in range(self.l_n)])
        self.pools = nn.ModuleList([GraphPool(ks[i], dim) for i in range(self.l_n)])
        self.unpools = nn.ModuleList([GraphUnpool() for i in range(self.l_n)])

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

            # Start Edit
            # X = torch.cat([X, down_outs[up_idx]], 1)
            # X = self.up_gcns[i](A, X)
            # End Edit

            # Start Before Edit
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
            # End Before Edit
        
        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)
        
        return X, start_gcn_outs[:, :268]
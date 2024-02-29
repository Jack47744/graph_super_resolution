import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class GCNLayer(nn.Module):
    """
    A single layer of a Graph Convolutional Network (GCN).
    ...
    """
    def __init__(self, input_dim, output_dim, use_nonlinearity=True):
        super(GCNLayer, self).__init__()
        self.use_nonlinearity = use_nonlinearity
        self.Omega = nn.Parameter(torch.randn(input_dim, output_dim) * torch.sqrt(torch.tensor(2.0) / (input_dim + output_dim)))
        self.beta = nn.Parameter(torch.zeros(output_dim))

    def forward(self, H_k, A_normalized):
        agg = torch.matmul(A_normalized, H_k)
        H_k_next = torch.matmul(agg, self.Omega) + self.beta
        return F.relu(H_k_next) if self.use_nonlinearity else H_k_next
    
class AEGraph(nn.Module):

    def __init__(self, low_dim, high_dim, hidden_dim):
        super(AEGraph, self).__init__()
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.hidden_dim = hidden_dim

        self.gcn1 = GCNLayer(self.hidden_dim, self.hidden_dim)
        self.gcn2 = GCNLayer(self.hidden_dim, self.hidden_dim)

    def forward(self, A_lr):

        H_0 = torch.eye(self.low_dim)

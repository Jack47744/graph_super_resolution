from pandas import period_range
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from initializations import *
from preprocessing import normalize_adj_torch

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

class GSRLayer(nn.Module):
  
  def __init__(self,lr_dim,hr_dim):
    super(GSRLayer, self).__init__()
    self.lr_dim = lr_dim
    self.hr_dim = hr_dim
    self.weights = torch.from_numpy(weight_variable_glorot(lr_dim*2)).type(torch.FloatTensor)
    self.weights = torch.nn.Parameter(data=self.weights, requires_grad = True)

  def forward(self,A,X):
    lr = A
    lr_dim = lr.shape[1]
    f = X
    eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U') 
    eye_mat = torch.eye(lr_dim).type(torch.FloatTensor)
    s_d = torch.cat((eye_mat, eye_mat),1).to(device)

    # print(f"GSRLayer s_d: {s_d.shape}")
    # print(f"GSRLayer U_lr: {U_lr.shape}")
    # print(f"GSRLayer weights: {self.weights.shape}")
    a = torch.matmul(self.weights, s_d.T)
    # print(f"GSRLayer a: {a.shape}")
    b = torch.matmul(a ,U_lr)
    # print(f"GSRLayer b: {b.shape}")
    # f_d = torch.matmul(b, f)
    # f_d_norm = torch.norm(f_d, dim=1)
    # _, f_d_idx = torch.topk(f_d_norm, self.hr_dim)
    # f_d = f_d[f_d_idx, :]

    # f_d = torch.matmul(b, f)
    # total_length = f_d.shape[0]
    # middle_length = self.hr_dim
    # start_index = (total_length - middle_length) // 2
    # end_index = start_index + middle_length
    # f_d = f_d[start_index:end_index, :]
    f_d = torch.matmul(b, f)[:self.hr_dim, :self.hr_dim]


    # print(f"GSRLayer f_d: {f_d.shape}")
    f_d = torch.abs(f_d)
    # self.f_d = f_d.fill_diagonal_(1)
    torch.diagonal(f_d, dim1=0, dim2=1).fill_(1) 
    self.f_d = f_d
    adj = normalize_adj_torch(self.f_d)
    # print(f"GSRLayer adj: {adj.shape}")
    X = adj @ adj.transpose(1, 2)
    X = (X + X.transpose(1,2))/2
    idx = torch.eye(self.hr_dim, dtype=bool)    
    # print(f"GSRLayer idx: {idx.shape}")
    # print(f"GSRLayer X: {X.shape}")
    X[:, idx]=1
    # print(adj.size(), X.size())
    return adj, torch.abs(X)
    


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    #160x320 320x320 =  160x320
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        # input = F.dropout(input, self.dropout, self.training)
        # print(f"GraphConvolution input: {input.shape}")
        # print(f"GraphConvolution weight: {self.weight.shape}")
        support = input @ self.weight
        # print(f"GraphConvolution support: {support.shape}")
        output = adj @ support
        # output = torch.mm(adj, support)
        # output = self.act(output)
        return output
    

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
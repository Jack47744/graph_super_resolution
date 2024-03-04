import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, Upsample
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
import numpy as np
from torch_geometric.data import Data
from torch.autograd import Variable


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

class Aligner(torch.nn.Module):
    def __init__(self, args):
        
        super(Aligner, self).__init__()

        self.args = args
        nn = Sequential(Linear(1, args.lr_dim*args.lr_dim), ReLU())
        self.conv1 = NNConv(args.lr_dim, args.lr_dim, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(args.lr_dim, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, args.lr_dim), ReLU())
        self.conv2 = NNConv(args.lr_dim, 1, nn, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, args.lr_dim), ReLU())
        self.conv3 = NNConv(1, args.lr_dim, nn, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(args.lr_dim, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, training=self.training)

        x3 = torch.cat([F.sigmoid(self.conv33(self.conv3(x2, edge_index, edge_attr))), x1], dim=1)
        x4 = x3[:, 0:self.args.lr_dim]
        x5 = x3[:, self.args.lr_dim:2*self.args.lr_dim]

        x6 = (x4 + x5) / 2
        return x6








class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        nn = Sequential(Linear(1, args.lr_dim*args.lr_dim),ReLU())
        self.conv1 = NNConv(args.lr_dim, args.lr_dim, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(args.lr_dim, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, args.hr_dim*args.lr_dim), ReLU())
        self.conv2 = NNConv(args.hr_dim, args.lr_dim, nn, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(args.lr_dim, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, args.hr_dim*args.lr_dim), ReLU())
        self.conv3 = NNConv(args.lr_dim, args.hr_dim, nn, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(args.hr_dim, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)


        # self.layer= torch.nn.ConvTranspose2d(N_TARGET_NODES, N_TARGET_NODES,5)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        # x = torch.squeeze(x)

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        # x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        # x2 = F.dropout(x2, training=self.training)

        x3 = F.sigmoid(self.conv33(self.conv3(x1, edge_index, edge_attr)))
        x3 = F.dropout(x3, training=self.training)



        x4  = torch.matmul(x3.t(), x3)

        return x4

class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.conv1 = GCNConv(args.hr_dim, args.hr_dim, cached=True)
        self.conv2 = GCNConv(args.hr_dim, 1, cached=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr
        x = torch.squeeze(x)
        x1 = F.sigmoid(self.conv1(x, edge_index))
        x1 = F.dropout(x1, training=self.training)
        x2 = F.sigmoid(self.conv2(x1, edge_index))
        #         # x2 = F.dropout(x2, training=self.training)


        return x2

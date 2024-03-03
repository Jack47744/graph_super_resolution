import torch
import torch.nn as nn
from layers import *
from ops import *
from preprocessing import normalize_adj_torch
import torch.nn.functional as F

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

class GSRNet(nn.Module):

  def __init__(self,ks,args):
    super(GSRNet, self).__init__()
    
    self.lr_dim = args.lr_dim
    self.hr_dim = args.hr_dim
    self.hidden_dim = args.hidden_dim
    self.layer = GSRLayer(self.lr_dim, self.hr_dim)
    self.net = GraphUnet(ks, self.lr_dim, self.hr_dim)
    self.gc1 = GraphConvolution(self.hr_dim, self.hidden_dim, 0, act=F.relu)
    self.gc2 = GraphConvolution(self.hidden_dim, self.hr_dim, 0, act=F.relu)

  def forward(self,lr):

    I = torch.eye(self.lr_dim).type(torch.FloatTensor).to(device)
    A = normalize_adj_torch(lr).type(torch.FloatTensor).to(device)

    self.net_outs, self.start_gcn_outs = self.net(A, I)
    
    self.outputs, self.Z = self.layer(A, self.net_outs)
    
    self.hidden1 = self.gc1(self.Z, self.outputs)
    self.hidden2 = self.gc2(self.hidden1, self.outputs)

    z = self.hidden2
    z = (z + z.t())/2
    idx = torch.eye(self.hr_dim, dtype=bool).to(device)
    z[idx] = 1
    
    # return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs
    return torch.relu(z), self.net_outs, self.start_gcn_outs, self.outputs
  
class Discriminator(nn.Module):
    """
    A simple Graph Neural Network model using two layers of Graph Convolutional Network (GCN)
    for binary classification. The sigmoid activation is applied in the output layer only if
    use_nonlinearity is set to True.
    """
    def __init__(self, input_dim, hidden_sizes=[], use_nonlinearity=True):
        super(Discriminator, self).__init__()
        self.use_nonlinearity = use_nonlinearity

        # Define GCN layers
        # self.gcn1 = GCNLayer(input_dim, hidden_dim_list[0], self.use_nonlinearity)
        # self.gcn2 = GCNLayer(hidden_dim_list[0], hidden_dim_list[1], self.use_nonlinearity)
        # self.gcn1 = GCNLayer(hidden_dim_list[1], 1, False)

        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(input_dim, hidden_sizes[0], self.use_nonlinearity))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(GCNLayer(hidden_sizes[i-1], hidden_sizes[i], self.use_nonlinearity))

        self.layers.append(GCNLayer(hidden_sizes[-1], 1, False))

    def forward(self, A, X):

        # Pass through GCN layers
        # H1 = self.gcn1(X, A)
        # H2 = self.gcn2(H1, A)

        for layer in self.layers[:-1]:
            X = layer(X, A)

        output = torch.sigmoid(X.mean(dim=0))

        return output
     
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
    elif torch.backends.mps.is_available():
        return torch.device("mps")
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
    # self.layer = GSRLayer(self.lr_dim, self.hr_dim)
    self.layer = GSRLayer(self.lr_dim, self.hidden_dim)
    self.net = GraphUnet(ks, self.lr_dim, self.hr_dim)
    self.gc1 = GraphConvolution(self.hr_dim, self.hidden_dim, 0, act=F.leaky_relu)
    self.gc2 = GraphConvolution(self.hidden_dim, self.hr_dim, 0, act=F.leaky_relu)

    self.gat1 = GAT(self.hr_dim, self.hidden_dim, F.leaky_relu)
    self.gat2 = GAT(self.hidden_dim, self.hr_dim, F.leaky_relu)


  def forward(self,lr):

    I = torch.eye(self.lr_dim).type(torch.FloatTensor).to(device)
    A = normalize_adj_torch(lr).type(torch.FloatTensor).to(device)

    self.net_outs, self.start_gcn_outs = self.net(A, I)
    
    self.outputs, self.Z = self.layer(A, self.net_outs)
    
    self.hidden1 = self.gc1(self.Z, self.outputs)
    self.hidden2 = self.gc2(self.hidden1, self.outputs)

    # self.hidden1 = self.gat1(self.outputs, self.Z)
    # self.hidden2 = self.gat2(self.outputs, self.hidden1)

    z = self.hidden2
    z = (z + z.t())/2
    idx = torch.eye(self.hr_dim, dtype=bool).to(device)
    z[idx] = 1
    
    # return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs
    return F.leaky_relu(z), self.net_outs, self.start_gcn_outs, self.outputs
  
# class Discriminator(nn.Module):
#     """
#     A simple Graph Neural Network model using two layers of Graph Convolutional Network (GCN)
#     for binary classification. The sigmoid activation is applied in the output layer only if
#     use_nonlinearity is set to True.
#     """
#     def __init__(self, input_dim, hidden_sizes=[], use_nonlinearity=True):
#         super(Discriminator, self).__init__()
#         self.use_nonlinearity = use_nonlinearity

#         # Define GCN layers
#         # self.gcn1 = GCNLayer(input_dim, hidden_dim_list[0], self.use_nonlinearity)
#         # self.gcn2 = GCNLayer(hidden_dim_list[0], hidden_dim_list[1], self.use_nonlinearity)
#         # self.gcn1 = GCNLayer(hidden_dim_list[1], 1, False)

#         self.layers = nn.ModuleList()
#         self.layers.append(GCNLayer(input_dim, hidden_sizes[0], self.use_nonlinearity))
#         for i in range(1, len(hidden_sizes)):
#             self.layers.append(GCNLayer(hidden_sizes[i-1], hidden_sizes[i], self.use_nonlinearity))

#         self.layers.append(GCNLayer(hidden_sizes[-1], 1, False))

#     def forward(self, A, X):

#         # Pass through GCN layers
#         # H1 = self.gcn1(X, A)
#         # H2 = self.gcn2(H1, A)

#         for layer in self.layers[:-1]:
#             X = layer(X, A)

#         output = torch.sigmoid(X.mean(dim=0))

#         return output

class Dense(nn.Module):
    def __init__(self, n1, n2, args):
        super(Dense, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.FloatTensor(n1, n2), requires_grad=True)
        nn.init.normal_(self.weights, mean=args.mean_dense, std=args.std_dense)

    def forward(self, x):
        np.random.seed(1)
        torch.manual_seed(1)

        out = torch.mm(x, self.weights)
        return out

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.dense_1 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_1 = nn.LeakyReLU(negative_slope=0.2)
        self.dense_2 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_2 = nn.LeakyReLU(negative_slope=0.2)
        self.dense_3 = Dense(args.hr_dim, 1, args)
        self.sigmoid = nn.Sigmoid()
        self.dropout_rate = args.dropout_rate

    def forward(self, x):
        # np.random.seed(1)
        # torch.manual_seed(1)
        x = F.dropout(self.relu_1(self.dense_1(x)), self.dropout_rate) + x
        x = F.dropout(self.relu_2(self.dense_2(x)), self.dropout_rate) + x
        x = self.dense_3(x)
        return self.sigmoid(x)
      
def gaussian_noise_layer(input_layer, args):
    z = torch.empty_like(input_layer)
    noise = z.normal_(mean=args.mean_gaussian, std=args.std_gaussian)
    z = torch.abs(input_layer + noise)

    z = (z + z.t())/2
    z = z.fill_diagonal_(1)
    return z
     
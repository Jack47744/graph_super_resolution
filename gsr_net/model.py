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
    I = I.unsqueeze(0).repeat(lr.size(0), 1, 1)
    A = normalize_adj_torch(lr).type(torch.FloatTensor).to(device)

    self.net_outs, self.start_gcn_outs = self.net(A, I)
    
    self.outputs, self.Z = self.layer(A, self.net_outs)
    
    self.hidden1 = self.gc1(self.Z, self.outputs)
    self.hidden2 = self.gc2(self.hidden1, self.outputs)

    z = self.hidden2
    # print(f"GSRNet z: {z.shape}")
    z = (z + z.transpose(1,2))/2
    idx = torch.eye(self.hr_dim, dtype=bool).to(device)
    z[:, idx] = 1
    
    # return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs
    return torch.relu(z), self.net_outs, self.start_gcn_outs, self.outputs
  
class Dense(nn.Module):
    def __init__(self, n1, n2, args):
        super(Dense, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.FloatTensor(n1, n2), requires_grad=True)
        nn.init.normal_(self.weights, mean=args.mean_dense, std=args.std_dense)

    def forward(self, x):
        np.random.seed(1)
        torch.manual_seed(1)

        # out = torch.mm(x, self.weights)
        out = x @ self.weights
        return out

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.dense_1 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_1 = nn.ReLU(inplace=False)
        self.dense_2 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_2 = nn.ReLU(inplace=False)
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
    
# class Discriminator(nn.Module):
#     def __init__(self, args):
#         super(Discriminator, self).__init__()

#         dim = (args.hr_dim**2 - args.hr_dim) // 2
#         # print(dim, args.hr_dim)

#         self.fc1 = nn.Linear(dim, dim//2)
#         self.fc2 = nn.Linear(dim//2, dim//4)
#         self.fc3 = nn.Linear(dim//4, dim//8)

#         # self.relu_1 = nn.ReLU(inplace=False)
#         # self.dense_2 = Dense(args.hr_dim, args.hr_dim, args)
#         # self.relu_2 = nn.ReLU(inplace=False)
#         # self.dense_3 = Dense(args.hr_dim, 1, args)
#         # self.sigmoid = nn.Sigmoid()
#         self.dropout_rate = 0.1

#     def forward(self, x):
#         # np.random.seed(1)
#         # torch.manual_seed(1)
#         # x = F.dropout(self.relu_1(self.dense_1(x)), self.dropout_rate)
#         # x = F.dropout(self.relu_2(self.dense_2(x)), self.dropout_rate)
#         # x = self.dense_3(x)
#         # return self.sigmoid(x)

#         # x = F.dropout()
#         x = F.dropout(F.relu(self.fc1(x)), self.dropout_rate)
#         x = F.dropout(F.relu(self.fc2(x)), self.dropout_rate)
#         x = self.fc3(x)

#         return F.sigmoid(x)


def gaussian_noise_layer(input_layer, args):
    # print(f"gaussian_noise_layer input_layer: {input_layer.shape}")
    z = torch.empty_like(input_layer)
    # print(f"gaussian_noise_layer z: {z.shape}")
    noise = z.normal_(mean=args.mean_gaussian, std=args.std_gaussian)
    z = torch.abs(input_layer + noise)
    # print(f"gaussian_noise_layer z: {z.shape}")
    z = (z + z.transpose(1,2))/2
    # z = z.fill_diagonal_(1)
    torch.diagonal(z, dim1=0, dim2=1).fill_(1) 
    return z
     
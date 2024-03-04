import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from model import *
import copy
import torch.optim as optim
from tqdm import tqdm
from losses import *


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
    
class CosineSimilarityAllLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityAllLoss, self).__init__()
    
    def forward(self, output, target):
        output_flat = output.view(output.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        cosine_loss = 1 - torch.mean(torch.cosine_similarity(output_flat, target_flat))
        return cosine_loss
    
class ColumnwiseCosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(ColumnwiseCosineSimilarityLoss, self).__init__()
    
    def forward(self, output, target):
        # Initialize cosine similarity function
        cosine_sim = nn.CosineSimilarity(dim=0)
        
        # Compute cosine similarity for each column and then take the mean
        # Assume output and target are square matrices of shape [n_nodes, n_nodes]
        cosine_sims = torch.tensor([cosine_sim(output[:, i], target[:, i]) for i in range(output.shape[1])])
        
        # Since we want to minimize the loss, and higher cosine similarity is better (closer to 1),
        # we subtract the mean similarity from 1 to represent a loss to minimize.
        cosine_loss = 1 - torch.mean(cosine_sims)
        return cosine_loss
    

# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss(beta=0.01)
criterion_L1 = nn.L1Loss()
kl_loss = nn.KLDivLoss()
bce_loss = nn.BCELoss()
cosine_sim_all_loss = CosineSimilarityAllLoss()
cosine_sim_col_loss = ColumnwiseCosineSimilarityLoss()

device = get_device()

def cal_laplacian(A):
    D = torch.diag(A.sum(1))
    L = D - A
    return L

def get_node_embedding(A, embedding_size):
    L = cal_laplacian(A)
    _, eigenvectors = torch.linalg.eigh(L)
    node_embeddings = eigenvectors[:, 1:embedding_size+1]
    return node_embeddings

def train_iman(
      netA,
      optimizerA,
      netG, 
      optimizerG,
      netD,
      optimizerD,
      subjects_adj, 
      subjects_labels, 
      args, 
      test_adj=None, 
      test_ground_truth=None
):
    netA.train()
    netG.train()
    netD.train()

    for epochs in range(args.epochs):
        # Train netG
        with torch.autograd.set_detect_anomaly(True):
            Al_losses = []

            Ge_losses = []
            losses_netD = []

            i = 0
            for data_source, data_target in zip(subjects_adj, subjects_labels):
                # ************    Domain alignment    ************
                A_output = netA(data_source)

                target = data_target.edge_attr.view(args.hr_dim, args.hr_dim).detach().cpu().clone().numpy()
                target_mean = np.mean(target)
                target_std = np.std(target)

                d_target = torch.normal(target_mean, target_std, size=(1, args.lr_dim_F))
                target_d = d_target.edge_attr.view(args.lr_dim, args.lr_dim)

                kl_loss = Alignment_loss(target_d, A_output)

                Al_losses.append(kl_loss)

                # ************     Super-resolution    ************
                G_output = netG(A_output)
                print("G_output: ", G_output.shape)
                G_output_reshaped = (G_output.view(1, args.hr_dim, args.hr_dim, 1).type(torch.FloatTensor)).detach()
                torch.cuda.empty_cache()

                Gg_loss = GT_loss(data_target, G_output)
                torch.cuda.empty_cache()
                D_real = netD(data_target)
                D_fake = netD(G_output_reshaped)
                torch.cuda.empty_cache()
                G_adversarial = adversarial_loss(D_fake, (torch.ones_like(D_fake, requires_grad=False)))
                G_loss = G_adversarial + Gg_loss
                Ge_losses.append(G_loss)

                D_real_loss = adversarial_loss(D_real, (torch.ones_like(D_real, requires_grad=False)))
                # torch.cuda.empty_cache()
                D_fake_loss = adversarial_loss(D_fake.detach(), torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2
                # torch.cuda.empty_cache()
                losses_netD.append(D_loss)
                i += 1

            # torch.cuda.empty_cache()

            optimizerG.zero_grad()
            Ge_losses = torch.mean(torch.stack(Ge_losses))
            Ge_losses.backward(retain_graph=True)
            optimizerG.step()

            optimizerA.zero_grad()
            Al_losses = torch.mean(torch.stack(Al_losses))
            Al_losses.backward(retain_graph=True)
            optimizerA.step()


            optimizerD.zero_grad()
            losses_netD = torch.mean(torch.stack(losses_netD))
            losses_netD.backward(retain_graph=True)
            optimizerD.step()

        print("[Epoch: %d]| [Al loss: %f]| [Ge loss: %f]| [D loss: %f]" % (epochs, Al_losses, Ge_losses, losses_netD))

    torch.save(netA.state_dict(), "./weight" + "netA_fold" + "_" + ".model")
    torch.save(netG.state_dict(), "./weight" + "netG_fold" + "_" + ".model")

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

def test(model, test_adj, test_labels,args):

  model.eval()
  test_error = []
  preds_list=[]
  g_t = []
  
  i=0
  # TESTING
  for lr, hr in zip(test_adj,test_labels):

    all_zeros_lr = not np.any(lr)
    all_zeros_hr = not np.any(hr)
    with torch.no_grad():
      if all_zeros_lr == False and all_zeros_hr==False: #choose representative subject
        lr = torch.from_numpy(lr).type(torch.FloatTensor)
        np.fill_diagonal(hr, 1)
        hr = torch.from_numpy(hr).type(torch.FloatTensor)
        preds,a,b,c = model(lr)
        # preds = unpad(preds, args.padding)

        #plot residuals
      #   if i==0:
      #     print ("Hr", hr)     
      #     print("Preds  ", preds)
      #     plt.imshow(hr, origin = 'upper',  extent = [-0.5, 268-0.5, 268-0.5, -0.5])
      #     plt.show(block=False)
      #     plt.imshow(preds.detach(), origin = 'upper',  extent = [-0.5, 268-0.5, 268-0.5, -0.5])
      #     plt.show(block=False)
      #     plt.imshow(hr - preds.detach(), origin = 'upper',  extent = [-0.5, 268-0.5, 268-0.5, -0.5])
      #     plt.show(block=False)
        
        preds_list.append(preds.flatten().detach().numpy())
        
        error = criterion_L1(preds, hr)
        g_t.append(hr.flatten())
        # print(error.item())
        test_error.append(error.item())
      
        i+=1
  # print ("Test error MSE: ", np.mean(test_error))
  return np.mean(test_error)
  
  #plot histograms
#   preds_list = [val for sublist in preds_list for val in sublist]
#   g_t_list = [val for sublist in g_t for val in sublist]
#   binwidth = 0.01
#   bins=np.arange(0, 1 + binwidth, binwidth)
#   plt.hist(preds_list, bins =bins,range=(0,1),alpha=0.5,rwidth=0.9, label='predictions')
#   plt.hist(g_t_list, bins=bins,range=(0,1),alpha=0.5,rwidth=0.9, label='ground truth')
#   plt.xlim(xmin=0, xmax = 1)
#   plt.legend(loc='upper right')
#   plt.title('GSR-Net with self reconstruction: Histogram')
#   plt.show(block=False)



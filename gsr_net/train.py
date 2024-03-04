import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from model import *
import copy
import torch.optim as optim
from tqdm import tqdm



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
    

def get_upper_triangle(matrix):
    n = matrix.shape[0]
    mask = torch.ones(n, n, dtype=torch.bool).triu().fill_diagonal_(False)
    return matrix[mask]

# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss(beta=0.01)
criterion_L1 = nn.L1Loss()
kl_loss = nn.KLDivLoss()
bce_loss = nn.BCELoss()
cosine_sim_all_loss = CosineSimilarityAllLoss()
cosine_sim_col_loss = ColumnwiseCosineSimilarityLoss()

device = get_device()


def train(model, optimizer, subjects_adj, subjects_labels, args, test_adj=None, test_ground_truth=None):
  
  all_epochs_loss = []
  no_epochs = args.epochs
  best_mae = np.inf
  early_stop_patient = args.early_stop_patient
  early_stop_count = 0
  best_model = None

  model = model.to(device)

  for epoch in tqdm(range(no_epochs), desc='Epoch Progress', unit='epoch'):

      epoch_loss = []
      epoch_error = []

      for lr,hr in zip(subjects_adj,subjects_labels):

          
          model.train()
          optimizer.zero_grad()
          
          lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
          hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)
          
          model_outputs, net_outs, start_gcn_outs, layer_outs = model(lr)

          padded_hr = pad_HR_adj(hr, args.padding).to(device)
          eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U') 

          mask = torch.ones_like(model_outputs, dtype=torch.bool)
          mask.fill_diagonal_(0)

          filtered_matrix1 = torch.masked_select(model_outputs, mask)
          filtered_matrix2 = torch.masked_select(hr, mask)
        

          loss = (
             args.lmbda * criterion(net_outs, start_gcn_outs) 
             + criterion(model.layer.weights, U_hr) 
             + criterion(filtered_matrix1, filtered_matrix2)
          )
          
          error = criterion_L1(model_outputs, hr)
          
          loss.backward()
          optimizer.step()

          epoch_loss.append(loss.item())
          epoch_error.append(error.item())
      
      all_epochs_loss.append(np.mean(epoch_loss))

      if test_adj is not None and test_ground_truth is not None:
        test_error = test(model, test_adj, test_ground_truth, args)


        if test_error < best_mae:
          best_mae = test_error
          early_stop_count = 0
          best_model = copy.deepcopy(model)
        elif early_stop_count >= early_stop_patient:
          if test_adj is not None and test_ground_truth is not None:
            test_error = test(best_model, test_adj, test_ground_truth, args)
            print(f"Val Error: {test_error:.6f}")
          return best_model
        else: 
          early_stop_count += 1

        tqdm.write(f'Epoch: {epoch+1}, Train Loss: {np.mean(epoch_loss):.6f}, '
               f'Train Error: {np.mean(epoch_error):.6f}, Test Error: {test_error:.6f}')
      else:
         tqdm.write(f'Epoch: {epoch+1}, Train Loss: {np.mean(epoch_loss):.6f}, '
               f'Train Error: {np.mean(epoch_error):.6f}')

  if not best_model:
      best_model = copy.deepcopy(model)

  if test_adj is not None and test_ground_truth is not None:
      test_error = test(model, test_adj, test_ground_truth, args)
      print(f"Val Error: {test_error:.6f}")

  return best_model

def cal_laplacian(A):
    D = torch.diag(A.sum(1))
    L = D - A
    return L

def get_node_embedding(A, embedding_size):
    L = cal_laplacian(A)
    _, eigenvectors = torch.linalg.eigh(L)
    node_embeddings = eigenvectors[:, 1:embedding_size+1]
    return node_embeddings

def train_gan(
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
  
  all_epochs_loss = []
  no_epochs = args.epochs
  best_mae = np.inf
  early_stop_patient = args.early_stop_patient
  early_stop_count = 0
  best_model = None

  netG = netG.to(device)
  netD = netD.to(device)

  for epoch in tqdm(range(no_epochs), desc='Epoch Progress', unit='epoch'):

      epoch_loss = []
      epoch_error = []

      for lr,hr in zip(subjects_adj,subjects_labels):

          
          netG.train()
          netD.train()
          optimizerG.zero_grad()
          optimizerD.zero_grad()
          
          lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
          hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)
          
          model_outputs, net_outs, start_gcn_outs, layer_outs = netG(lr)

          padded_hr = pad_HR_adj(hr, args.padding).to(device)
          _, U_hr = torch.linalg.eigh(padded_hr, UPLO='U') 

          mask = torch.ones_like(model_outputs, dtype=torch.bool)
          mask.fill_diagonal_(0)

          filtered_matrix1 = torch.masked_select(model_outputs, mask)
          filtered_matrix2 = torch.masked_select(hr, mask)

          mse_loss = (
             args.lmbda * criterion(net_outs, start_gcn_outs) 
             + criterion(netG.layer.weights, U_hr) 
             + criterion(filtered_matrix1, filtered_matrix2)
          )
          
          # Discriminator Update

          error = criterion_L1(model_outputs, hr)
          real_data = model_outputs.detach()
          
          total_length = padded_hr.shape[0]
          middle_length = args.hr_dim
          start_index = (total_length - middle_length) // 2
          end_index = start_index + middle_length
          padded_hr = padded_hr[start_index:end_index, start_index:end_index]


          fake_data = gaussian_noise_layer(padded_hr, args)

          # d_real = netD(get_upper_triangle(real_data))
          # d_fake = netD(get_upper_triangle(fake_data))

          d_real = netD(real_data)
          d_fake = netD(fake_data)

          dc_loss_real = bce_loss(d_real, torch.ones_like(d_real))
          dc_loss_fake = bce_loss(d_fake, torch.zeros_like(d_real))
          dc_loss = dc_loss_real + dc_loss_fake

          dc_loss.backward()
          optimizerD.step()

          # Generator Update

          # d_fake = netD(get_upper_triangle(gaussian_noise_layer(padded_hr, args)))
          d_fake = netD(gaussian_noise_layer(padded_hr, args))

          gen_loss = bce_loss(d_fake, torch.ones_like(d_fake))
          generator_loss = gen_loss + mse_loss
          generator_loss.backward()
          optimizerG.step()

          epoch_loss.append(generator_loss.item())
          epoch_error.append(error.item())
      
      all_epochs_loss.append(np.mean(epoch_loss))

      if test_adj is not None and test_ground_truth is not None:
        test_error = test(netG, test_adj, test_ground_truth, args)


        if test_error < best_mae:
          best_mae = test_error
          early_stop_count = 0
          best_model = copy.deepcopy(netG)
        elif early_stop_count >= early_stop_patient:
          if test_adj is not None and test_ground_truth is not None:
            test_error = test(best_model, test_adj, test_ground_truth, args)
            print(f"Val Error: {test_error:.6f}")
          return best_model
        else: 
          early_stop_count += 1

        tqdm.write(f'Epoch: {epoch+1}, Train Loss: {np.mean(epoch_loss):.6f}, '
               f'Train Error: {np.mean(epoch_error):.6f}, Test Error: {test_error:.6f}')
      else:
         tqdm.write(f'Epoch: {epoch+1}, Train Loss: {np.mean(epoch_loss):.6f}, '
               f'Train Error: {np.mean(epoch_error):.6f}')

  if not best_model:
      best_model = copy.deepcopy(netG)

  if test_adj is not None and test_ground_truth is not None:
      test_error = test(netG, test_adj, test_ground_truth, args)
      print(f"Val Error: {test_error:.6f}")

  return best_model
    
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


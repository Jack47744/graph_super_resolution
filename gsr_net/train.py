import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from model import *
import copy
import torch.optim as optim



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

  for epoch in range(no_epochs):

      epoch_loss = []
      epoch_error = []

      for lr,hr in zip(subjects_adj,subjects_labels):

          
          model.train()
          optimizer.zero_grad()
          
          lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
          hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)
          
          model_outputs, net_outs, start_gcn_outs, layer_outs = model(lr)
          # model_outputs  = unpad(model_outputs, args.padding)

          padded_hr = pad_HR_adj(hr, args.padding).to(device)
          eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U') 
          # print(net_outs.size(),start_gcn_outs.size())
          # print(model.layer.weights.size(), U_hr.size())
          # print(model_outputs.size(), hr.size())

          mask = torch.ones_like(model_outputs, dtype=torch.bool)
          mask.fill_diagonal_(0)

          filtered_matrix1 = torch.masked_select(model_outputs, mask)
          filtered_matrix2 = torch.masked_select(hr, mask)
        

          loss = (
             args.lmbda * criterion(net_outs, start_gcn_outs) 
             + criterion(model.layer.weights, U_hr) 
             + criterion(filtered_matrix1, filtered_matrix2)
            #  + cosine_sim_col_loss(model.layer.weights, U_hr) 
          )
          
          error = criterion_L1(model_outputs, hr)
          # error_L1 = criterion_L1(model_outputs, hr)
          
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

        print(f"Epoch: {epoch+1}, Train Loss: {np.mean(epoch_loss):.6f}, Train Error: {np.mean(epoch_error):.6f}, Test Error: {test_error:.6f}")
        # print("Epoch: ",i, "Train Loss: ", np.mean(epoch_loss), "Train Error: ", np.mean(epoch_error),", Test Error: ", test_error)
      else:
        print(f"Epoch: {epoch+1}, Train Loss: {np.mean(epoch_loss):.6f}, Train Error: {np.mean(epoch_error):.6f}")

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
    early_stop_patient = 3
    early_stop_count = 0
    best_model = None

    
    # netD = Discriminator(input_dim = 256, hidden_dim_list=[256, 128])
    # optimizerD = optim.Adam(netD.parameters(), lr=args.lr)
    bce_loss = nn.BCELoss()

    netG = netG.to(device)
    netD = netD.to(device)

    for epoch in range(no_epochs):

        epoch_loss = []
        epoch_error = []

        for lr, hr in zip(subjects_adj, subjects_labels):
            
            # Prep Data
            lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
            hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)
            padded_hr = pad_HR_adj(hr, args.padding).to(device)
            _, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')
            U_hr = U_hr.to(device)

      
            hr_node_embeddings = get_node_embedding(hr, args.embedding_size).to(device)
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.train()
            optimizerD.zero_grad()
            
            output = netD(hr, hr_node_embeddings).view(-1)
            errD_real = bce_loss(output, torch.ones_like(output).to(device))
            errD_real.backward()

            fake, net_outs, start_gcn_outs, layer_outs = netG(lr)
            fake_hr_node_embeddings = get_node_embedding(fake, args.embedding_size).to(device)

            output = netD(hr, fake_hr_node_embeddings.detach()).view(-1)
            errD_fake = bce_loss(output, torch.zeros_like(output).to(device))
            errD_fake.backward()
            
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.train()
            optimizerG.zero_grad()
            output = netD(hr, fake_hr_node_embeddings).view(-1)
            errG = bce_loss(output, torch.ones_like(output).to(device))
            
            loss = (
              args.lmbda * criterion(net_outs, start_gcn_outs) 
              + criterion(netG.layer.weights, U_hr) 
              + criterion(fake, hr)
              + errG
            )

            loss.backward()
            optimizerG.step()
            
            error = criterion_L1(fake, hr)

            epoch_loss.append(loss.item())
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
            return netG
          else: 
            early_stop_count += 1

          train_loss = f"{np.mean(epoch_loss):.6f}"
          train_error = f"{np.mean(epoch_error):.6}"
          test_error = f"{test_error:.6f}"

          print(
              f"Epoch: {epoch+1}, Train Loss: {train_loss}, Train Error: {train_error},  Test Error: {test_error}"
          )
        else:
          print(f"Epoch: {epoch+1}, Train Loss: {np.mean(epoch_loss):.6f}, Train Error: {train_error}")

    if not best_model:
        best_model = netG

    if test_adj is not None and test_ground_truth is not None:
        test_error = test(netG, test_adj, test_ground_truth, args)
        print(f"Val Error: {test_error:.6f}")

    if best_model:
        return best_model
    return netG

#   plt.plot(all_epochs_loss)
#   plt.title('GSR-UNet with self reconstruction: Loss')
#   plt.show(block=False)
    
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



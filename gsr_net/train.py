import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from model import *
import copy

criterion = nn.MSELoss()
criterion_L1 = nn.L1Loss()
kl_loss = nn.KLDivLoss()

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


def train(model, optimizer, subjects_adj, subjects_labels, args, test_adj=None, test_ground_truth=None):
  
  all_epochs_loss = []
  no_epochs = args.epochs
  best_mae = np.inf
  early_stop_patient = 3
  early_stop_count = 0
  best_model = None

  for epoch in range(no_epochs):

      epoch_loss = []
      epoch_error = []

      for lr,hr in zip(subjects_adj,subjects_labels):

          model = model.to(device)
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
          
          loss = (
             args.lmbda * criterion(net_outs, start_gcn_outs) 
             + criterion(model.layer.weights, U_hr) 
             + criterion(model_outputs, hr)
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
          return best_model
        else: 
          early_stop_count += 1

        print(f"Epoch: {epoch}, Train Loss: {np.mean(epoch_loss):.6f}, Train Error: {np.mean(epoch_error):.6f}, Test Error: {test_error:.6f}")
        # print("Epoch: ",i, "Train Loss: ", np.mean(epoch_loss), "Train Error: ", np.mean(epoch_error),", Test Error: ", test_error)
      else:
        print(f"Epoch: {epoch}, Train Loss: {np.mean(epoch_loss):.6f}, Train Error: {np.mean(epoch_error):.6f}")

  if best_model:
    return best_model
  return model

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
        np.fill_diagonal(hr,1)
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



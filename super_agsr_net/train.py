import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import pad_HR_adj
from model import gaussian_noise_layer
import copy
from tqdm.notebook import tqdm
from utils import get_device


device = get_device()

criterion = nn.SmoothL1Loss(beta=0.01)
criterion_L1 = nn.L1Loss()
bce_loss = nn.BCELoss()


def cal_error(model_outputs, hr, mask):
    return criterion_L1(model_outputs[mask], hr[mask])


def train_gan(
    netG,
    optimizerG,
    netD,
    optimizerD,
    subjects_adj,
    subjects_labels,
    args,
    test_adj=None,
    test_ground_truth=None,
    stop_gan_mae=None,
):
    """
    Train the GAN AGSR model function
    
    :param netG: The generator model
    :param optimizerG: The optimizer for the generator
    :param netD: The discriminator model
    :param optimizerD: The optimizer for the discriminator
    :param subjects_adj: The adjacency matrices of the subjects
    :param subjects_labels: The labels of the subjects
    :param args: The arguments for the model
    :param test_adj: The adjacency matrices of the test subjects
    :param test_ground_truth: The ground truth labels of the test subjects
    :param stop_gan_mae: The mean absolute error to stop the GAN training
    :return: The trained GAN model"""

    all_epochs_loss = []
    no_epochs = args.epochs  # Number of epochs for training
    best_mae = np.inf  # Initialize best mean absolute error
    early_stop_patient = args.early_stop_patient  # Early stopping patience
    early_stop_count = 0  # Early stopping counter
    best_model = None  # Initialize best model

    # Move models to the appropriate device
    netG = netG.to(device)
    netD = netD.to(device)

    # Create a mask for upper triangle
    mask = torch.triu(torch.ones(args.hr_dim, args.hr_dim), diagonal=1).bool()

    # Progress bar for epochs
    with tqdm(range(no_epochs), desc="Epoch Progress", unit="epoch") as tepoch:

        for epoch in tepoch:
            epoch_loss = []
            epoch_error = []

            # Set models to training mode
            netG.train()
            netD.train()

            # Iterate over subjects
            for lr, hr in zip(subjects_adj, subjects_labels):
                # Zero the gradients
                optimizerG.zero_grad()
                optimizerD.zero_grad()

                # Convert numpy arrays to tensors and move to device
                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)

                # Forward pass through the generator
                model_outputs, net_outs, start_gcn_outs, layer_outs = netG(lr)

                # Pad the high-resolution adjacency matrix and eigen-decomposition
                padded_hr = pad_HR_adj(hr, args.padding).to(device)
                _, U_hr = torch.linalg.eigh(padded_hr, UPLO="U")
                U_hr = U_hr.to(device)
                
                # Move outputs to the device
                model_outputs = model_outputs.to(device)
                net_outs = net_outs.to(device)
                start_gcn_outs = start_gcn_outs.to(device)

                # Update mask for current output dimensions
                mask = torch.ones_like(model_outputs, dtype=torch.bool).to(device)
                mask.fill_diagonal_(0)

                # Apply mask to filter the matrices
                filtered_matrix1 = torch.masked_select(model_outputs, mask)
                filtered_matrix2 = torch.masked_select(hr, mask)

                # Calculate the composite loss
                mse_loss = (
                    args.lmbda * criterion(net_outs, start_gcn_outs)
                    + criterion(netG.layer.weights, U_hr)
                    + criterion(filtered_matrix1, filtered_matrix2)
                )

                # Calculate error between generated and real data
                error = cal_error(model_outputs, hr, mask)
                real_data = model_outputs.detach()

                # Preprocess HR matrix for discriminator
                total_length = padded_hr.shape[0]
                middle_length = args.hr_dim
                start_index = (total_length - middle_length) // 2
                end_index = start_index + middle_length
                padded_hr = padded_hr[start_index:end_index, start_index:end_index]

                # Train discriminator if condition met
                if stop_gan_mae is None or best_mae >= stop_gan_mae:
                    fake_data = gaussian_noise_layer(padded_hr, args)

                    d_real = netD(real_data)
                    d_fake = netD(fake_data)

                    dc_loss_real = bce_loss(d_real, torch.ones_like(d_real))
                    dc_loss_fake = bce_loss(d_fake, torch.zeros_like(d_real))
                    dc_loss = dc_loss_real + dc_loss_fake

                    dc_loss.backward()
                    optimizerD.step()

                # Update generator
                if stop_gan_mae is None or best_mae >= stop_gan_mae:
                    d_fake = netD(gaussian_noise_layer(padded_hr, args))

                    gen_loss = bce_loss(d_fake, torch.ones_like(d_fake))
                    generator_loss = gen_loss + mse_loss
                else:
                    generator_loss = criterion(filtered_matrix1, filtered_matrix2)

                generator_loss.backward()
                optimizerG.step()

                epoch_loss.append(generator_loss.item())
                epoch_error.append(error.item())

            all_epochs_loss.append(np.mean(epoch_loss))

            # Evaluate on test data if provided
            if test_adj is not None and test_ground_truth is not None:
                test_error = test_gan(netG, test_adj, test_ground_truth, args)

                if test_error < best_mae:
                    best_mae = test_error
                    early_stop_count = 0
                    best_model = copy.deepcopy(netG)
                elif early_stop_count >= early_stop_patient:
                    # Early stopping condition met
                    if test_adj is not None and test_ground_truth is not None:
                        test_error = test_gan(
                            best_model, test_adj, test_ground_truth, args
                        )
                        print(f"Val Error: {test_error:.6f}")
                    return best_model
                else:
                    early_stop_count += 1

                tepoch.set_postfix(
                    train_loss=np.mean(epoch_loss),
                    train_error=np.mean(epoch_error),
                    test_error=test_error,
                )
            else:
                tepoch.set_postfix(
                    train_loss=np.mean(epoch_loss), train_error=np.mean(epoch_error)
                )

    if not best_model:
        best_model = copy.deepcopy(netG)

    # Final evaluation on test data if available
    if test_adj is not None and test_ground_truth is not None:
        test_error = test_gan(netG, test_adj, test_ground_truth, args)
        print(f"Val Error: {test_error:.6f}")

    return best_model

def test_gan(model, test_adj, test_labels, args):
    """
    Test the GAN AGSR model function

    :param model: The trained GAN model
    :param test_adj: The adjacency matrices of the test subjects
    :param test_labels: The labels of the test subjects
    :param args: The arguments for the model
    :return: The mean absolute error of the model on the test data
    """

    model.eval()
    test_error = []
    g_t = []

    mask = (
        torch.triu(torch.ones(args.hr_dim, args.hr_dim), diagonal=1).bool().to(device)
    )

    i = 0
    # TESTING
    for lr, hr in zip(test_adj, test_labels):

        all_zeros_lr = not np.any(lr)
        all_zeros_hr = not np.any(hr)
        with torch.no_grad():
            if (
                all_zeros_lr == False and all_zeros_hr == False
            ):  # choose representative subject
                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                np.fill_diagonal(hr, 1)
                hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)
                preds, _, _, _ = model(lr)
                preds = preds.to(device)

                error = cal_error(preds, hr, mask)
                g_t.append(hr.flatten())

                test_error.append(error.item())

                i += 1
    # print ("Test error MSE: ", np.mean(test_error))
    return np.mean(test_error)

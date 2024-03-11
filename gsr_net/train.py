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

    print("inside Training GAN")
    all_epochs_loss = []
    no_epochs = args.epochs
    best_mae = np.inf
    early_stop_patient = args.early_stop_patient
    early_stop_count = 0
    best_model = None

    netG = netG.to(device)
    netD = netD.to(device)

    mask = torch.triu(torch.ones(args.hr_dim, args.hr_dim), diagonal=1).bool()

    with tqdm(range(no_epochs), desc="Epoch Progress", unit="epoch") as tepoch:

        for epoch in tepoch:

            epoch_loss = []
            epoch_error = []

            netG.train()
            netD.train()

            for lr, hr in zip(subjects_adj, subjects_labels):

                optimizerG.zero_grad()
                optimizerD.zero_grad()

                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)

                model_outputs, net_outs, start_gcn_outs, layer_outs = netG(lr)

                padded_hr = pad_HR_adj(hr, args.padding).to(device)
                _, U_hr = torch.linalg.eigh(padded_hr, UPLO="U")
                U_hr = U_hr.to(device)
                model_outputs = model_outputs.to(device)
                net_outs = net_outs.to(device)
                start_gcn_outs = start_gcn_outs.to(device)

                mask = torch.ones_like(model_outputs, dtype=torch.bool).to(device)
                mask.fill_diagonal_(0)

                filtered_matrix1 = torch.masked_select(model_outputs, mask)
                filtered_matrix2 = torch.masked_select(hr, mask)

                mse_loss = (
                    args.lmbda * criterion(net_outs, start_gcn_outs)
                    + criterion(netG.layer.weights, U_hr)
                    + criterion(filtered_matrix1, filtered_matrix2)
                )

                error = cal_error(model_outputs, hr, mask)
                real_data = model_outputs.detach()

                total_length = padded_hr.shape[0]
                middle_length = args.hr_dim
                start_index = (total_length - middle_length) // 2
                end_index = start_index + middle_length
                padded_hr = padded_hr[start_index:end_index, start_index:end_index]

                if stop_gan_mae is None or best_mae >= stop_gan_mae:
                    fake_data = gaussian_noise_layer(padded_hr, args)

                    d_real = netD(real_data)
                    d_fake = netD(fake_data)

                    dc_loss_real = bce_loss(d_real, torch.ones_like(d_real))
                    dc_loss_fake = bce_loss(d_fake, torch.zeros_like(d_real))
                    dc_loss = dc_loss_real + dc_loss_fake

                    dc_loss.backward()
                    optimizerD.step()

                # Generator Update

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

            if test_adj is not None and test_ground_truth is not None:
                test_error = test_gan(netG, test_adj, test_ground_truth, args)

                if test_error < best_mae:
                    best_mae = test_error
                    early_stop_count = 0
                    best_model = copy.deepcopy(netG)
                elif early_stop_count >= early_stop_patient:
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

    if test_adj is not None and test_ground_truth is not None:
        test_error = test_gan(netG, test_adj, test_ground_truth, args)
        print(f"Val Error: {test_error:.6f}")

    return best_model


def test_gan(model, test_adj, test_labels, args):
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

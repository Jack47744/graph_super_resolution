import torch
import numpy as np
import argparse
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from MatrixVectorizer import MatrixVectorizer
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
import psutil
import os

process = psutil.Process(os.getpid())

LR_size = 160
HR_size = 268
EPOCHS = 200


def track_memory():
    """
    Track the memory usage of the process
    """
    ram_usage = process.memory_info().rss / (1024**2)  # Convert to MB
    print(f"Current memory usage: {ram_usage:.2f} MB")


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


def weight_variable_glorot(output_dim):
    """
    Initialize weights using the Glorot uniform initialization method.

    Parameters:
    - output_dim: The number of output features of the layer.
    
    Returns:
    - A NumPy array of weights initialized using the Glorot uniform method.
    """

    input_dim = output_dim
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = np.random.uniform(-init_range, init_range, (input_dim, output_dim))

    return initial


def compute_degree_matrix_normalization_batch_numpy(adjacency_batch):
    """
    Optimizes the degree matrix normalization for a batch of adjacency matrices using NumPy.
    Computes the normalized adjacency matrix D^-1 * A for each graph in the batch.

    Parameters:
    - adjacency_batch: A NumPy array of shape (batch_size, num_nodes, num_nodes) representing
                    a batch of adjacency matrices.

    Returns:
    - A NumPy array of normalized adjacency matrices.
    """
    epsilon = 1e-6  # Small constant to avoid division by zero
    # Calculate the degree for each node in the batch
    d = adjacency_batch.sum(axis=2) + epsilon

    # Compute the inverse degree matrix D^-1 for the batch
    D_inv = (
        np.reciprocal(d)[:, :, np.newaxis]
        * np.eye(adjacency_batch.shape[1])[np.newaxis, :, :]
    )

    # Normalize the adjacency matrix using batch matrix multiplication
    normalized_adjacency_batch = np.matmul(D_inv, adjacency_batch)

    return normalized_adjacency_batch


def get_parser():
    """
    Create an argument parser for the GSR-Net model.

    Returns:
    - An argument parser for the GSR-Net model.
    """

    parser = argparse.ArgumentParser(description="GSR-Net")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        metavar="no_epochs",
        help="number of episode to train ",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00010436239086997339,
        metavar="lr",
        help="learning rate (default: 0.0001 using Adam Optimizer)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=3,
        metavar="n_splits",
        help="no of cross validation folds",
    )
    parser.add_argument(
        "--lmbda",
        type=int,
        default=24,
        metavar="L",
        help="self-reconstruction error hyperparameter",
    )
    parser.add_argument(
        "--lr_dim",
        type=int,
        default=LR_size,
        metavar="N",
        help="adjacency matrix input dimensions",
    )
    parser.add_argument(
        "--hr_dim",
        type=int,
        default=HR_size,
        metavar="N",
        help="super-resolved adjacency matrix output dimensions",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=268,
        metavar="N",
        help="hidden GraphConvolutional layer dimensions",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=26,
        metavar="padding",
        help="dimensions of padding",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=32,
        metavar="embedding_size",
        help="node embedding size",
    )
    parser.add_argument(
        "--early_stop_patient",
        type=int,
        default=5,
        metavar="early_stop_patient",
        help="early_stop_patience",
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        metavar="dropout_rate",
        help="dropout_rate",
    )
    parser.add_argument(
        "--mean_dense",
        type=float,
        default=0.0,
        metavar="mean",
        help="mean of the normal distribution in Dense Layer",
    )
    parser.add_argument(
        "--std_dense",
        type=float,
        default=0.01,
        metavar="std",
        help="standard deviation of the normal distribution in Dense Layer",
    )
    parser.add_argument(
        "--mean_gaussian",
        type=float,
        default=0.0,
        metavar="mean",
        help="mean of the normal distribution in Gaussian Noise Layer",
    )
    parser.add_argument(
        "--std_gaussian",
        type=float,
        default=0.1,
        metavar="std",
        help="standard deviation of the normal distribution in Gaussian Noise Layer",
    )

    parser.add_argument(
        "--ks",
        type=list,
        default=[0.9, 0.7, 0.6, 0.5],
        metavar="ks",
        help="ks",
    )

    return parser


def evaluate(pred_matrices, gt_matrices, cal_graph=False):
    """
    Evaluate the performance of the model using the mean absolute error (MAE)
    and Pearson correlation coefficient (PCC).

    Parameters:
    - pred_matrices: A PyTorch tensor of predicted adjacency matrices.
    - gt_matrices: A PyTorch tensor of ground truth adjacency matrices.
    - cal_graph: A boolean indicating whether to compute graph centrality measures.
    
    Returns:
    - A dictionary containing the evaluation metrics.
    """

    # pred_matrices = pred_matrices.cpu().detach().numpy()
    # gt_matrices = gt_matrices.cpu().detach().numpy()

    num_test_samples = gt_matrices.shape[0]

    # Initialize lists to store MAEs for each centrality measure
    mae_bc = []
    mae_ec = []
    mae_pc = []

    pred_1d = []
    gt_1d = []

    # Iterate over each test sample
    for i in tqdm(range(num_test_samples)):

        pred_1d.append(MatrixVectorizer.vectorize(pred_matrices[i]))
        gt_1d.append(MatrixVectorizer.vectorize(gt_matrices[i]))

        if cal_graph:
            # Convert adjacency matrices to NetworkX graphs
            pred_graph = nx.from_numpy_array(pred_matrices[i])
            gt_graph = nx.from_numpy_array(gt_matrices[i])

            # Compute centrality measures
            pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
            pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight")
            pred_pc = nx.pagerank(pred_graph, weight="weight")

            gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
            gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight")
            gt_pc = nx.pagerank(gt_graph, weight="weight")

            # Convert centrality dictionaries to lists
            pred_bc_values = list(pred_bc.values())
            pred_ec_values = list(pred_ec.values())
            pred_pc_values = list(pred_pc.values())

            gt_bc_values = list(gt_bc.values())
            gt_ec_values = list(gt_ec.values())
            gt_pc_values = list(gt_pc.values())

            # Compute MAEs
            mae_bc.append(mean_absolute_error(pred_bc_values, gt_bc_values))
            mae_ec.append(mean_absolute_error(pred_ec_values, gt_ec_values))
            mae_pc.append(mean_absolute_error(pred_pc_values, gt_pc_values))

    if cal_graph:
        # Compute average MAEs
        avg_mae_bc = sum(mae_bc) / len(mae_bc)
        avg_mae_ec = sum(mae_ec) / len(mae_ec)
        avg_mae_pc = sum(mae_pc) / len(mae_pc)

    # vectorize and flatten
    pred_1d = np.concatenate(pred_1d, axis=0).flatten()
    gt_1d = np.concatenate(gt_1d, axis=0).flatten()

    mae = mean_absolute_error(pred_1d, gt_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    js_dis = jensenshannon(pred_1d, gt_1d)

    print("MAE: ", mae)
    print("PCC: ", pcc)
    print("Jensen-Shannon Distance: ", js_dis)
    if cal_graph:
        print("Average MAE betweenness centrality:", avg_mae_bc)
        print("Average MAE eigenvector centrality:", avg_mae_ec)
        print("Average MAE PageRank centrality:", avg_mae_pc)

    if cal_graph:

        res = {
            "MAE": mae,
            "PCC": pcc,
            "JSD": js_dis,
            "MAE_(BC)": avg_mae_bc,
            "MAE_(EC)": avg_mae_ec,
            "MAE_(PC)": avg_mae_pc,
        }
    else:
        res = {
            "MAE": mae,
            "PCC": pcc,
            "JSD": js_dis,
            # "MAE_(BC)": avg_mae_bc,
            # "MAE_(EC)": avg_mae_ec,
            # "MAE_(PC)": avg_mae_pc
        }

    return res


def plot_metrics_fold(res_list):
    df = pd.DataFrame(res_list)
    df = df.rename(
        columns={
            "mae": "MAE",
            "pcc": "PCC",
            "js_dis": "JSD",
            "avg_mae_bc": "MAE_(BC)",
            "avg_mae_ec": "MAE_(EC)",
            "avg_mae_pc": "MAE_(PC)",
        }
    )
    df.index = df.index.set_names(["Fold"])
    df.loc["mean"] = df.mean()
    avg_data = df.iloc[-1, :]
    df.loc["std"] = df.std()
    errors = df.iloc[-1, :].tolist()
    df = df.reset_index()
    df = df.iloc[:-2, :]
    df_long = df.melt(id_vars="Fold", var_name="Metric", value_name="Value")
    palette = sns.color_palette("muted", n_colors=len(df_long["Metric"].unique()))
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for fold in range(3):
        i = fold // 2
        j = fold % 2
        sns.barplot(
            x="Metric",
            y="Value",
            data=df_long[df_long["Fold"] == fold],
            ax=axs[i, j],
            palette=palette,
            hue="Metric",
        )
        axs[i, j].set_title(f"Fold: {fold+1}")

    bars = sns.barplot(
        x=avg_data.index, y=avg_data.values, ax=axs[1, 1], palette=palette, capsize=5
    )

    for i, bar in enumerate(bars.patches):
        # Calculate the x-position of the error bar
        x_pos = bar.get_x() + bar.get_width() / 2

        # Add error bar for this metric
        axs[1, 1].errorbar(
            x_pos,
            avg_data.values[i],
            yerr=errors[i],
            fmt="none",
            color="black",
            capsize=5,
        )

    axs[1, 1].set_title("Avg. Across Folds")
    plt.show()

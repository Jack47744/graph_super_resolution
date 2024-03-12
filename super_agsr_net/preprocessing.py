import torch
import torch.nn.functional as F
from utils import get_device
from MatrixVectorizer import MatrixVectorizer

device = get_device()


def pad_HR_adj(label, split):
    """
    Pad the adjacency matrix of the HR graph with zeros
    :param label: The adjacency matrix of the HR graph
    :param split: The number of zeros to pad the matrix with
    :return: The padded adjacency matrix
    """

    # Pad the tensor
    padding = (split, split, split, split)  # Padding for left, right, top, bottom
    label_padded = F.pad(label, padding, "constant", 0)

    # Create an identity matrix of the same size as the padded tensor
    identity = torch.eye(label_padded.size(0)).to(device)

    # Add the identity matrix to the padded tensor to set diagonal elements to 1
    # Assuming the operation intended is to ensure diagonal elements are set to 1 post padding
    label_padded = label_padded + identity

    return label_padded.type(torch.FloatTensor)


def normalize_adj_torch(mx):
    """
    Normalize the adjacency matrix
    :param mx: The adjacency matrix
    :return: The normalized adjacency matrix
    """

    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx


def antivectorize_df(adj_mtx_df, size):
    num_subject = adj_mtx_df.shape[0]
    adj_mtx = np.zeros(
        (num_subject, size, size)
    )  # torch.zeros((num_subject, LR_size, LR_size))
    for i in range(num_subject):
        adj_mtx[i] = MatrixVectorizer.anti_vectorize(adj_mtx_df.iloc[i], size)
    return adj_mtx

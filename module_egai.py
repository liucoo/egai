import torch
import torch.nn as nn
import numpy as np
import math


def generate_new_trainset(thre, output, labels, idx_train):
    y_pse = labels.cpu().detach().numpy().copy()
    preds = nn.Softmax(dim=1)(output).cpu().detach().numpy()

    train_list_seg = [
        i for i in range(labels.shape[0])
        if i not in set(idx_train.detach().numpy()) and np.max(preds[i]) > thre
    ]
    y_pse[train_list_seg] = np.argmax(preds[train_list_seg], axis=1)
    return y_pse, train_list_seg


def total_ssl(edges_list, list_nbunch, y_pse, output, ratio):
    s_ = set(list_nbunch)
    edges_candi_list = [i for i in edges_list if i[0] in s_ and i[1] in s_]

    embd_mat = np.dot(output.cpu().detach().numpy(),
                      output.cpu().detach().numpy().transpose())

    remove_list_inter = [
        i for i in edges_candi_list if y_pse[i[0]] != y_pse[i[1]]
    ]

    s = set(remove_list_inter)
    remove_list_intra = [x for x in edges_candi_list if x not in s]
    similarity_list = [embd_mat[i[0], i[1]] for i in remove_list_intra]
    similarity_list_sorted = np.argsort(0 - np.array(similarity_list))
    remove_list_final = []
    for rank in range(int(ratio * len(remove_list_intra))):
        remove_list_final.append(
            remove_list_intra[similarity_list_sorted[rank, ]])

    remove_list_total = remove_list_final + remove_list_inter

    G_remove = G_.copy()
    G_remove.remove_edges_from(remove_list_total)
    adj_mod = nx.to_numpy_array(G_remove)
    adj_normalizer = fetch_normalization("NormAdj")
    adj_mod = sparse_mx_to_torch_sparse_tensor(adj_normalizer(adj_mod))

    return adj_mod
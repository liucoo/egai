import numpy as np
import scipy.sparse as sp
import sys
import pickle as pkl
import networkx as nx
import math
from time import perf_counter
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim

from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
from module_egai import generate_new_trainset ,total_ssl

# Training settings
lr = 0.01
weight_decay = 5e-4
hidden = 16
dropout = 0.5
dataset = "cora"
model_name = "GCN"
feature = "mul"  # choices=['mul', 'cat', 'adj'],
normalization = "NormAdj"
per = -1
cuda = False


adj_0, adj, features, labels, idx_train, idx_val, idx_test = load_citation(
    dataset, normalization, cuda)

G_ = nx.from_numpy_array(adj_0.toarray())
edges = G_.edges()
edges_list = [i for i in edges]

def train_regression(model,train_features,train_labels,val_features,val_labels, dropout,adj_aug):
    model.train()
    optimizer.zero_grad()
    output = model(train_features, adj_aug)
    loss_train = F.cross_entropy(output[idx_train], train_labels)
    loss_train.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        output_val = model(val_features, adj)
        loss_val = F.cross_entropy(output_val[idx_val], val_labels)

    return model, loss_val, output

def test_regression(model, test_features, idx_test, test_labels):
    model.eval()
    output = model(test_features, adj)
    return accuracy(output[idx_test], test_labels)



# train

percent_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]
ratio_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
result_array = np.zeros((8, 9, 10))

train_list  = [int(i) for i in idx_train]
for seed_index, seed in enumerate(range(42, 52)):

    for per_index, percent in enumerate(percent_list):
        for ratio_index, ratio in enumerate(ratio_list):
            set_seed(seed, cuda)
            print("Percent: {} Ratio: {} Seed:{} ".
                      format(percent, ratio, seed))
            model = get_model("GCN", features.size(1),
                  labels.max().item() + 1, hidden, dropout, cuda)
            optimizer = torch.optim.Adam([
        dict(params=model.gc1.parameters(), weight_decay=5e-4),
        dict(params=model.gc2.parameters(), weight_decay=0)
    ],
                                 lr=0.01)

            cost_val = []
            early_stopping = 10

            for epoch in range(200):
                if epoch < 1:
                    adj_aug = adj
                else:
                    adj_aug = total_ssl(edges_list, list_nbunch, y_pse, output,
                                        ratio)
                model, loss_val, output = train_regression(
                    model, features, labels[idx_train], features,
                    labels[idx_val], dropout, adj_aug)
                cost_val.append(loss_val)

                if epoch > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping+1):-1]):
                    print("Early stopping...")
                    break


                y_pse, train_list_seg = generate_new_trainset(
                    percent, output, labels, idx_train)
                list_nbunch = train_list_seg + train_list

            acc_test = test_regression(model, features, idx_test,
                                           labels[idx_test])
            result_array[per_index, ratio_index, seed_index] = acc_test

            print(" Test Accuracy: {:.4f} ".
                      format( acc_test))


import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import time
from collections import defaultdict
from torch.utils.data import DataLoader
import networkx as nx
import matplotlib.pyplot as plt

from dataset import GraphDataset
from utils.preprocessing import *
import args
import model

def get_scores(edges_pos, edges_neg, adj_rec, adj_orig):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

if __name__=='__main__':
    # Train on CPU (hide GPU) due to memory constraints
    # os.environ['CUDA_VISIBLE_DEVICES'] = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # init model and optimizer
    model = getattr(model, args.model)(device).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # 创建 Dataset
    graph_dataset = GraphDataset(args)
    # 使用 DataLoader 加载数据
    dataloader = DataLoader(graph_dataset, batch_size=1, shuffle=True)

    all_train_loss = []
    all_train_acc = []
    # all_val_roc = []
    # all_val_ap = []
    # train model
    all_t = time.time()
    for epoch in range(args.num_epoch):
        t = time.time()
        # 遍历 DataLoader
        tot_loss = 0
        tot_train_acc = 0
        # tot_val_roc = 0
        # tot_val_ap = 0
        for feature, adjacency_matrix in dataloader:
            feature = feature.numpy().reshape(-1, args.input_dim)
            adjacency_matrix = adjacency_matrix.numpy().reshape(adjacency_matrix.shape[1], -1)
            graph = defaultdict(dict)
            # 遍历邻接矩阵的每个元素，将非零元素加入 defaultdict
            for i in range(adjacency_matrix.shape[0]):
                for j in range(adjacency_matrix.shape[1]):
                    weight = adjacency_matrix[i, j]
                    if weight != 0:
                        graph[i][j] = {'weight': weight}
            # Output the adjacency matrix size after the loop
            adj = nx.adjacency_matrix(nx.from_dict_of_dicts(graph), weight='weight')
            feature = sp.csr_matrix(feature).tolil()
            # Store original adjacency matrix (without diagonal entries) for later
            adj_orig = adj
            adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
            adj_orig.eliminate_zeros()

            # adj_train, train_edges, val_edges, val_edges_false = mask_test_edges(adj)
            # adj = adj_train

            # print(adj)
            # Some preprocessing
            adj_norm = preprocess_graph(adj)

            num_nodes = adj.shape[0]

            feature = sparse_to_tuple(feature.tocoo())
            num_feature = feature[2][1]
            feature_nonzero = feature[1].shape[0]

            # Create Model
            adj_edge_num = np.sum(adj != 0)
            pos_weight = float(adj.shape[0] * adj.shape[0] - adj_edge_num) / adj_edge_num
            norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj_edge_num) * 2)

            # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            # norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
            # print(pos_weight)
            # print(norm)
            # adj_label = adj + sp.eye(adj.shape[0])
            adj_label = sparse_to_tuple(adj)
            # adj_label = adj_norm

            adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                                torch.FloatTensor(adj_norm[1]),
                                                torch.Size(adj_norm[2])).to(device)
            adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                                 torch.FloatTensor(adj_label[1]),
                                                 torch.Size(adj_label[2])).to(device)
            feature = torch.sparse.FloatTensor(torch.LongTensor(feature[0].T),
                                               torch.FloatTensor(feature[1]),
                                               torch.Size(feature[2])).to(device)

            # weight_mask = adj_label.to_dense().view(-1) == 1
            weight_mask = adj_label.to_dense().view(-1) != 0
            weight_tensor = torch.ones(weight_mask.size(0)).to(device)
            weight_tensor[weight_mask] = pos_weight
            # feature, adj_norm, norm, adj_label, weight_tensor, val_edges, val_edges_false, adj_orig
            A_pred, Z = model(feature, adj_norm)
            optimizer.zero_grad()
            loss = log_lik = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
            # loss = log_lik = norm * F.mse_loss(A_pred.view(-1), adj_label.to_dense().view(-1))
            # print(adj_label.to_dense().view(-1))
            # print(torch.max(adj_label.to_dense().view(-1)), torch.min(adj_label.to_dense().view(-1)))
            # print(A_pred.view(-1))
            if args.model == 'VGAE':
                kl_divergence = 0.5 / A_pred.size(0) * (
                        1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()
                loss -= kl_divergence

            loss.backward()
            optimizer.step()

            train_acc = get_acc(A_pred, adj_label).cpu()

            # val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred.cpu(), adj_orig)

            tot_loss += loss.item()
            tot_train_acc += train_acc
            # tot_val_roc += val_roc
            # tot_val_ap += val_ap

        all_train_loss.append(tot_loss / len(dataloader))
        all_train_acc.append(tot_train_acc / len(dataloader))
        # all_val_roc.append(tot_val_roc / len(dataloader))
        # all_val_ap.append(tot_val_ap / len(dataloader))

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(tot_loss/len(dataloader)),
              "train_acc=", "{:.5f}".format(tot_train_acc/len(dataloader)),
              "time=", "{:.5f}".format(time.time() - t))
        #  "val_roc=", "{:.5f}".format(tot_val_roc/len(dataloader)),
        #                        "val_ap=", "{:.5f}".format(tot_val_ap/len(dataloader)),

    # test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    # print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
    #       "test_ap=", "{:.5f}".format(test_ap))
    print("train time:" + str(time.time()-all_t))
    epochs = range(1, args.num_epoch + 1)
    plt.plot(epochs, all_train_loss, label='Train Loss')
    plt.plot(epochs, all_train_acc, label='Train Accuracy')
    # plt.plot(epochs, all_val_roc, label='Validation ROC')
    # plt.plot(epochs, all_val_ap, label='Validation AP')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig('./figure/{}_metric.png'.format(args.model))
    # 保存整个模型
    torch.save(model.state_dict(), './trained_model/{}_model.pth'.format(args.model))
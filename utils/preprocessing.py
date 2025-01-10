import numpy as np
import scipy.sparse as sp

# 获取矩阵的行列，矩阵值，形状
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    # 将输入的邻接矩阵 adj 转换为 COO 格式的稀疏矩阵
    adj = sp.coo_matrix(adj)
    # 给图中的每个节点添加自环，以保证每个节点都与自身相连
    # adj_ = adj + sp.eye(adj.shape[0])
    adj_ = adj
    # 计算邻接矩阵加上自环后的行和，然后计算每个节点度的倒数平方根。这个倒数平方根的计算用于后续对邻接矩阵的归一化操作。
    rowsum = np.array((adj_ != 0).sum(1))
    # rowsum = np.array((adj_ != 0).sum(axis=1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # 归一化邻接矩阵：通过左右乘以度的倒数平方根，对邻接矩阵进行对称归一化
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def preprocess_graph2(adj):
    # 将输入的邻接矩阵 adj 转换为 COO 格式的稀疏矩阵
    adj = sp.coo_matrix(adj)
    x = adj.data
    adj_normalized = np.exp(-x) * 0.1
    adj_normalized = sp.coo_matrix((adj_normalized, (adj.row, adj.col)), shape=adj.shape)
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    # 删去矩阵中的对角元素，由于对角元素是结点与其自身之间的连接
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  # 取原始矩阵的上三角矩阵
    adj_tuple = sparse_to_tuple(adj_triu)
    # edges是只取了上三角矩阵的边，所以边无重复
    edges = adj_tuple[0]
    # edges_all取了矩阵所有的边
    edges_all = sparse_to_tuple(adj)[0]
    # num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    # test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    # test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, val_edge_idx, axis=0) # np.hstack([val_edge_idx])

    # 检查数组a中的任意一行是否在数组b中
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        # if ismember([idx_i, idx_j], test_edges):
        #     continue
        # if ismember([idx_j, idx_i], test_edges):
        #     continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false  #, test_edges, test_edges_false
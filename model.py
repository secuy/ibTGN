import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import args

class VGAE(nn.Module):
	def __init__(self, device=None):
		super(VGAE,self).__init__()
		self.device = device
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, activation=lambda x:x)

	def encode(self, X, adj):
		hidden = self.base_gcn(X, adj)
		self.mean = self.gcn_mean(hidden, adj)
		self.logstd = self.gcn_logstddev(hidden, adj)
		gaussian_noise = torch.randn(X.size(0), args.hidden2_dim).to(self.device)
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X, adj):
		Z = self.encode(X, adj)
		A_pred = dot_product_decode(Z)
		return A_pred, Z

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim)
		self.activation = activation

	def forward(self, inputs, adj):
		x = inputs
		x = torch.mm(x, self.weight)
		x = torch.mm(adj, x)
		outputs = self.activation(x)
		return outputs

# 点积可以捕捉到节点之间的相似性，而 sigmoid 函数的使用可以将得分映射到一个概率值，
# 表示节点之间是否存在边。这个过程可以被视为一种图的重构，即尝试在学习的节点表示的基础上，通过内积关系还原出图的拓扑结构。
def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

# 用于初始化权重矩阵，采用 Glorot 初始化，以确保良好的训练性能。
def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)


class GAE(nn.Module):
	def __init__(self, device=None):
		super(GAE, self).__init__()
		self.device = device
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, activation=lambda x:x)

	def encode(self, X, adj):
		hidden = self.base_gcn(X, adj)
		z = self.mean = self.gcn_mean(hidden, adj)
		return z

	def forward(self, X, adj):
		Z = self.encode(X, adj)
		A_pred = dot_product_decode(Z)
		return A_pred, Z

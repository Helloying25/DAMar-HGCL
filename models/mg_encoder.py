import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Mg_encoder(nn.Module):
    def __init__(self, hidden_dim, mp_len, dropout, attn_drop=0.0):
        super(Mg_encoder, self).__init__()

        self.non_linear = nn.ReLU()
        self.overall_graph_gen = GraphChannelAttLayer(mp_len, attn_drop=attn_drop)
        self.feat_drop = nn.Dropout(dropout)
        self.GCN = GraphConvolution(hidden_dim, hidden_dim)

    def forward(self, feature, MetaGraph, L_norm):
        new_G = self.overall_graph_gen(MetaGraph)
        new_G = F.normalize(new_G, dim=1, p=L_norm)
        feature = self.feat_drop(feature)
        h = F.tanh(self.GCN(feature, new_G))
        return h, new_G


class GraphChannelAttLayer(nn.Module):
    def __init__(self, num_channel, weights=None, attn_drop=0.0):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)
        if weights is not None:
            self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))
        if attn_drop > 0:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        a = F.softmax(self.attn_drop(self.weight), dim=0)
        return torch.sum(adj_list * a, dim=0)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output
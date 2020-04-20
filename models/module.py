import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
import dgl
from dgl import DGLGraph
import dgl.function as fn
import argparse
import pickle
#from GraphBuilder import Graph
from tqdm import tqdm
import numpy as np
import math
from IPython import embed

class NCE_HINGE(nn.Module):
    """docstring for NCE_HINGE"""
    def __init__(self, margin=1):
        super(NCE_HINGE, self).__init__()
        self.margin = margin

    def forward(self, scores, others=None):
        #print(scores.shape)
        return torch.sum(F.relu(scores[:, 0].unsqueeze(1) - scores[:, 1:] + self.margin)) / scores.shape[0] + torch.sum(F.relu(scores[:, 0] - 1)) / scores.shape[0]

class BatchPairwiseDistance(nn.Module):
    r"""
    Computes the batchwise pairwise distance between vectors :math:`v_1`, :math:`v_2` using the p-norm:

    .. math ::
        \Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}.

    Args:
        p (real): the norm degree. Default: 2
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-6
        keepdim (bool, optional): Determines whether or not to keep the vector dimension.
            Default: False
    Shape:
        - Input1: :math:`(N, D)` where `D = vector dimension`
        - Input2: :math:`(N, D)`, same shape as the Input1
        - Output: :math:`(N)`. If :attr:`keepdim` is ``True``, then :math:`(N, 1)`.
    Examples::
        >>> pdist = nn.PairwiseDistance(p=2)
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> output = pdist(input1, input2)
    """
    __constants__ = ['norm', 'eps', 'keepdim']

    def __init__(self, p=2., eps=1e-6, keepdim=False):
        super(BatchPairwiseDistance, self).__init__()
        self.norm = p
        self.eps = eps
        self.keepdim = keepdim

    #@weak_script_method
    def forward(self, x1, x2):
        results = torch.cat((x1[:, 0, :].unsqueeze(1) - x2, x2[:, 0, :].unsqueeze(1) - x1[:, 1:, :]), 1)
        return results.norm(p=self.norm, dim=2)

class RelEdgeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, params = None, attn_param = None, attn = False):
        super(RelEdgeUpdate, self).__init__()
        if params is None:
            self.linear = nn.Linear(in_feats, out_feats, bias = True)
        else:
            self.linear = params
        self.cross_attn = attn
        if not self.cross_attn:
            if attn_param:
                self.attn_fc = attn_param
            else:
                self.attn_fc = nn.Linear(2 * 
                    out_feats, 1, bias=True)

    def forward(self, edges):
        if self.cross_attn:
            return {'m': self.linear(edges.src['h'])}
        else:
            z1 = self.linear(edges.src['h'])
            z2 = torch.cat([z1, edges.dst['self_h']], dim=1)
            a = self.attn_fc(z2)
            return {'z': z1, 'e': F.leaky_relu(a)}

class smallGraphAlignLayer(nn.Module):
    def __init__(self,
                in_feats,
                n_hidden,
                activation,
                dropout,
                num_rels
                ):
        super(smallGraphAlignLayer, self).__init__()

        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.rel_layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.num_rels = num_rels
        self.self_attn = nn.Linear(2 * n_hidden, 1, bias=False)

        for i in range(2*num_rels+1):
            #projection_layer
            layer = nn.Linear(in_feats, n_hidden, bias = True)
            self.rel_layers.append(RelEdgeUpdate(in_feats, n_hidden, params = layer, attn_param = self.self_attn, attn = False))
            self.attn_layers.append(RelEdgeUpdate(in_feats, n_hidden, params = layer, attn_param = self.self_attn, attn = True))

        self.activation = activation


    def reduce(self, node):
        if 'm' in node.mailbox:
        #if False:
            mask = node.mailbox['m'].sum(dim=2) != 0
            attn_weights = torch.exp(-torch.norm(node.mailbox['z'][:, :, None] - node.mailbox['m'][:, None], dim=3, p=2))
            tmp_attn = attn_weights * (1 - mask[:,:,None].float()) * mask[:,None].float()
            batch_weight = tmp_attn.sum(dim=2) / tmp_attn.sum(dim=2).sum(dim=1)[:,None]
            alpha = F.softmax(node.mailbox['e'], dim=1)
            batch_weight = batch_weight * alpha.squeeze()
            batch_weight = batch_weight / batch_weight.sum(dim=1)[:,None]
            return {'z': torch.cat([node.data['h'], torch.sum(node.mailbox['z']*batch_weight[:,:,None], dim=1)], dim=1)}
        else:
            return {'z': torch.cat([node.data['h'], node.mailbox['z'].mean(dim=1)], dim=1)}
        #return {'z': torch.cat([node.data['h'], node.mailbox['z']], dim=0)}

    def cross_attn(self, edges):
        return {''}

    def forward(self, g, feat, edge_indices, node_indices = None, attn = False):
        g = g.local_var()
        g.ndata['h'] = feat
        g.ndata['self_h'] = self.rel_layers[0].linear(g.ndata['h'])
        # bi-directional relations
        for i in range(2 * self.num_rels):
            g.send(edge_indices[i+1], self.rel_layers[i+1])

        if attn:
            for i in range(2 * self.num_rels):
                g.send(edge_indices[-i-1], self.attn_layers[i+1])

        # self-loop
        # print(g.ndata['h'].shape, self.rel_layers[0].linear.weight.shape)
        g.ndata['h'] = g.ndata['self_h']
        # attention edges
        # g.send_recv
        if node_indices is None:
            g.recv(g.nodes(), reduce_func = self.reduce, apply_node_func = lambda node : {'activation': self.activation(node.data['z'])})
        else:
            g.recv(node_indices, reduce_func = self.reduce, apply_node_func = lambda node : {'activation': self.activation(node.data['z'])})
        #print('here')
        return g.ndata['activation']

class smallGraphAlignNet(nn.Module):
    """Subgraph Alignment Network for Entity Linkage
        model option:
            1. 1-Layer RGCN
            2. 2-Layer RGCN
            3. 2-Layer RGCN + parameter_sharing 
            4. 2-Layer RGCN + cross attention
            5. 1-Layer RGCN + parameter_sharing + cross attention
            5. 2-Layer RGCN + parameter_sharing + cross attention
    """
    def __init__(self,
                in_feats,
                num_neighbors,
                num_negatives,
                n_hidden,
                n_layers,
                activation,
                dropout,
                num_rel_1,
                num_rel_2,
                model_opt,
                dist,
                loss_fcn
                ):
        super(smallGraphAlignNet, self).__init__()

        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        self.layers.append(smallGraphAlignLayer(in_feats, n_hidden, activation, dropout, num_rel_1))
        
        for i in range(n_layers):
            self.layers.append(smallGraphAlignLayer(2*n_hidden, n_hidden, activation, dropout, num_rel_1))

        # self.g = num_neighbors
        self.dist = dist
        self.fc = nn.Linear(2*n_hidden, 1)
        self.loss_fcn = loss_fcn

    def predict(self, emb, train_ids, batch_size, num_negatives, n_hidden, offset):
        loss = 0.0
        for batch in tdata.DataLoader(train_ids, batch_size=batch_size*(num_negatives+1), shuffle=False):
        #print(batch.shape)
            output_a, output_b = emb[batch[:, 0]].view(-1, num_negatives+1, 2 * n_hidden), emb[batch[:, 1] + offset].view(-1, num_negatives+1, 2 * n_hidden)
            logits = self.dist(output_a, output_b)
            loss += self.loss_fcn(logits)
        return loss
    
    def forward(self, g, edge_indices, node_indices = None):
        self.g = g
        h = self.g.ndata['features']
        for i, layer in enumerate(self.layers):
            #print('here')
            if i != len(self.layers) - 1:
                h = layer(self.g, h, edge_indices, node_indices[len(self.layers) - 1 - i])
            else:
                h = layer(self.g, h, edge_indices, node_indices[0], attn = True)

        return h
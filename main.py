from models import module
#import utils
import argparse, pickle, time
from IPython import embed
from preprocess import *
import torch.utils.data as tdata
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from dgl import DGLGraph
from tqdm import tqdm
from torch.nn.modules.distance import PairwiseDistance
import itertools
#from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import random
from sklearn.metrics import roc_auc_score, f1_score


def genEdgeBatch(g, train_data, graph_a, graph_b, adj_a, adj_b, type_a_dict, type_b_dict, add_edge = True, num_hops = 1, num_neighbors = 10):
    train_data = train_data.numpy()
    nodes_a, nodes_b = set(train_data[:, 0].tolist()), set(train_data[:, 1].tolist())

    nodes = [list(nodes_a) + list(map(lambda x:x+len(graph_a.id2idx), nodes_b))]

    edge_indices = defaultdict(list)
    eids = []

    left_nodes, right_nodes = set(), set()

    if True:
        for i in range(train_data.shape[0]):
            #left_nodes.add(train_data[i, 0])
            #right_nodes.add(train_data[i, 1])
            for n in random.sample(adj_a[train_data[i, 0]], min(num_neighbors, len(adj_a[train_data[i, 0]]))):
                left_nodes.add(n)
                for sub_edge in type_a_dict[(n, train_data[i,0])]:
                    edge_indices[sub_edge + 1].append(g.edge_id(n, train_data[i,0]))
                if add_edge:
                    g.add_edge(n, train_data[i, 1]+len(graph_a.id2idx))

                e_id = g.edge_id(n, train_data[i, 1]+len(graph_a.id2idx))
                #attn_edges.append(-type_a_dict[(n, train_data[i,0])] - 1)
                for sub_edge in type_a_dict[(n, train_data[i,0])]:
                    edge_indices[-sub_edge - 1].append(e_id)
                eids.append(e_id)
            for m in random.sample(adj_b[train_data[i, 1]], min(num_neighbors, len(adj_b[train_data[i, 1]]))):
                right_nodes.add(m)
                for sub_edge in type_b_dict[(m, train_data[i,1])]:
                    edge_indices[sub_edge + 1].append(g.edge_id(m+len(graph_a.id2idx), train_data[i,1]+len(graph_a.id2idx)))
                if add_edge:
                    g.add_edge(m+len(graph_a.id2idx), train_data[i, 0])
                    # here is duplicate
                e_id = g.edge_id(m+len(graph_a.id2idx), train_data[i, 0])

                #attn_edges.append(-type_b_dict[(m, train_data[i,1])] - 1)
                for sub_edge in type_b_dict[(m, train_data[i,1])]:
                    edge_indices[-sub_edge - 1].append(e_id)
                eids.append(e_id)
    #embed()
    if num_hops > 1:
    #if False:
        nodes.append(list(left_nodes) + list(map(lambda x:x+len(graph_a.id2idx), right_nodes)))
        for node_id in list(left_nodes):
            for n in random.sample(adj_a[node_id], min(num_neighbors, len(adj_a[node_id])) ):
                for sub_edge in type_a_dict[(n, node_id)]:
                    try:
                        edge_indices[sub_edge + 1].append(g.edge_id(n, node_id))
                    except:
                        embed()
        for node_id in list(right_nodes):
            for m in random.sample(adj_b[node_id], min(num_neighbors, len(adj_b[node_id])) ):
                for sub_edge in type_b_dict[(m, node_id)]:
                    edge_indices[sub_edge + 1].append(g.edge_id(m+len(graph_a.id2idx), node_id+len(graph_a.id2idx)))      
    #embed()
    #assert len(eids) == len(set(eids))
    return edge_indices, nodes, eids

# Sub-sample a K-hop graph for small graph entity linkage
def genSubGraph(graph_a, graph_b, num_hops=1):
    #nodes_a, nodes_b = set(train_data[:, 0].tolist()), set(train_data[:, 1].tolist())
    #print(len(nodes_a), len(nodes_b))
    #edge_indices = defaultdict(list)
    g = DGLGraph()
    g.add_nodes(len(graph_a.id2idx) + len(graph_b.id2idx))
    
    g.add_edges(graph_a.edge_src, graph_a.edge_dst)
    g.add_edges(graph_a.edge_dst, graph_a.edge_src)
    
    #g.add_edges(list(range(len(graph_a.id2idx))), list(range(len(graph_a.id2idx))))

    #offset
    g.add_edges(list(map(lambda x:x+len(graph_a.id2idx), graph_b.edge_src)), list(map(lambda x:x+len(graph_a.id2idx), graph_b.edge_dst)))
    g.add_edges(list(map(lambda x:x+len(graph_a.id2idx), graph_b.edge_dst)), list(map(lambda x:x+len(graph_a.id2idx), graph_b.edge_src)))
    #g.add_edges(list(range(len(graph_a.id2idx), len(graph_a.id2idx)+len(graph_b.id2idx))), 
     #   list(range(len(graph_a.id2idx), len(graph_a.id2idx)+len(graph_b.id2idx))))

    edge_type_a, edge_type_b = torch.LongTensor(graph_a.edge_type), torch.LongTensor(graph_b.edge_type)
    num_type_a, num_type_b = torch.max(edge_type_a).item() + 1, torch.max(edge_type_b).item() + 1
    type_a_dict, type_b_dict = defaultdict(list), defaultdict(list)
    adj_a, adj_b = defaultdict(list), defaultdict(list)
    for a,b,t in zip(graph_a.edge_src, graph_a.edge_dst, graph_a.edge_type):
        if b not in adj_a[a]:
            adj_a[a].append(b)
        if a not in adj_a[b]:
            adj_a[b].append(a)
        type_a_dict[(a,b)].append(t)
        type_a_dict[(b,a)].append(t + num_type_a)

    for a,b,t in zip(graph_b.edge_src, graph_b.edge_dst, graph_b.edge_type):
        if b not in adj_b[a]:
            adj_b[a].append(b)
        if a not in adj_b[b]:
            adj_b[b].append(a)
        type_b_dict[(a,b)].append(t)
        type_b_dict[(b,a)].append(t + num_type_b)

    # print(num_type_a, num_type_b)
    # assume same number of relations 
    assert num_type_a == num_type_b
    
    num_edges = g.number_of_edges()

    # concatenating two graphs
    g.ndata['features'] = torch.cat([torch.FloatTensor(graph_a.features), torch.FloatTensor(graph_b.features)], 0).cuda()


    return g, num_type_a, len(graph_a.id2idx), adj_a, adj_b, type_a_dict, type_b_dict

def mergeGraph(graph_a, graph_b, train_data):
    g = DGLGraph()
    g.add_nodes(len(graph_a.id2idx) + len(graph_b.id2idx))
    
    g.add_edges(graph_a.edge_src, graph_a.edge_dst)
    g.add_edges(graph_a.edge_dst, graph_a.edge_src)
    

    #offset
    g.add_edges(list(map(lambda x:x+len(graph_a.id2idx), graph_b.edge_src)), list(map(lambda x:x+len(graph_a.id2idx), graph_b.edge_dst)))
    g.add_edges(list(map(lambda x:x+len(graph_a.id2idx), graph_b.edge_dst)), list(map(lambda x:x+len(graph_a.id2idx), graph_b.edge_src)))

    print(train_data.shape)

    print(g.number_of_edges())
    for i in range(train_data.shape[0]):
        g.add_edge(train_data[i, 0], train_data[i, 1] + len(graph_a.id2idx))
        g.add_edge(train_data[i, 1] + len(graph_a.id2idx), train_data[i, 0])
    
    num_edges = g.number_of_edges()
    g.ndata['features'] = torch.cat([torch.FloatTensor(graph_a.features), torch.FloatTensor(graph_b.features)], 0).cuda()

    return g


def main(args):
    graph_a, graph_b = Graph(args.pretrain_path), Graph(args.pretrain_path)
    graph_a.buildGraph('data/itunes_amazon_exp_data/exp_data/tableA.csv')
    graph_b.buildGraph('data/itunes_amazon_exp_data/exp_data/tableB.csv')
    # embed()
    train_data, val_data, test_data = generateTrainWithType('data/itunes_amazon_exp_data/exp_data/', graph_a, graph_b, positive_only=args.model_opt==0)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
    #print('here')
    g, num_rel, offset, adj_a, adj_b, type_a_dict, type_b_dict = genSubGraph(graph_a, graph_b, args.n_layers+1)
    
    in_feats = g.ndata['features'].shape[1]

    if args.model_opt == 0:
        loss_fcn = module.NCE_HINGE()
    else:
        loss_fcn = nn.BCEWithLogitsLoss()

    model = module.BatchPairwiseDistance(p=2)
    if args.gat == False:
        model_gan = module.smallGraphAlignNet(in_feats,
            g,
            args.num_negatives,
            args.n_hidden,
            args.n_layers,
            F.relu,
            args.dropout,
            num_rel,
            num_rel,
            args.model_opt,
            dist=model,
            loss_fcn=loss_fcn
            )



    if cuda:
        #g = g.cuda()
        model_gan.cuda()
        model.cuda()

    optimizer = torch.optim.Adam([{'params': model_gan.parameters()}],
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    if args.validation:
        writer = SummaryWriter(comment=args.model_id + 'person_type')
        writer1 = SummaryWriter(comment=args.model_id + 'film_type')

    
    print(model_gan)
    #test_id = torch.LongTensor(test_id)
    train_loader = tdata.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = tdata.DataLoader(test_data, batch_size=train_data.shape[0], shuffle=False)
    val_loader = tdata.DataLoader(val_data, batch_size=val_data.shape[0], shuffle=False)

    #writer.add_graph(model_gan, [edge_indices, torch.LongTensor(train_ids), args.batch_size, args.num_negatives, args.n_hidden, offset])
    best_roc_score = 0
    for epoch in range(args.n_epochs):
        model_gan.train()
        model.train()
        training_loss = 0.0
        eids = []
        for batch in train_loader:
            # two graphs are concatenated
            test_edges, test_nodes, eid = genEdgeBatch(g, batch, graph_a, graph_b, adj_a, adj_b, type_a_dict, type_b_dict, num_hops = args.n_layers + 1, num_neighbors = args.num_neighbors)
            #print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
            eids += eid
            emb = model_gan(g, test_edges, test_nodes)
            # embed()
            if False:
                output_a, output_b = emb[batch[:, 0]].view(-1, args.num_negatives+1, 2 * args.n_hidden), emb[batch[:, 1] + offset].view(-1, args.num_test_negatives+1, 2 * args.n_hidden)
                #g.remove_edges(eid)
                logits = model(output_a, output_b)
                loss = loss_fcn(logits)
            else:
                # embed()
                # emb = g.ndata['features']
                loss = loss_fcn( model_gan.fc(emb[batch[:, 0]]*emb[batch[:, 1]+ offset]).squeeze(), batch[:, 2].cuda().float() )
            training_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        g.remove_edges(eids)
        del emb
        torch.cuda.empty_cache()
        
        print('Epoch:{}, loss:{}'.format(epoch, training_loss ))

        with torch.no_grad():
            eids = []
            for batch in val_loader:
                # two graphs are concatenated
                test_edges, test_nodes, eid = genEdgeBatch(g, batch, graph_a, graph_b, adj_a, adj_b, type_a_dict, type_b_dict, num_hops = args.n_layers + 1, num_neighbors = args.num_neighbors)
                #print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
                eids += eid
                emb = model_gan(g, test_edges, test_nodes)
                # emb = g.ndata['features']
                score = model_gan.fc(emb[batch[:, 0]]*emb[batch[:, 1]+ offset]).squeeze() #.sum(dim=1)
                roc_score = roc_auc_score(batch[:,2].numpy(), score.detach().cpu().numpy())
                best_f1 = 0
                for i in range(10):
                    f1 = f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i ) 
                    best_f1 = max(best_f1, f1)
                    #print('ths:{}, f1:{}'.format(i, f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i )))
                # embed()
                print('Validation AUC_ROC:{}, Best F1:{}'.format(roc_score, best_f1))
                if roc_score > best_roc_score:
                    torch.save(model_gan.state_dict(), 'best_gan.pkl')
            g.remove_edges(eids)
        
    model_gan.load_state_dict(torch.load('best_gan.pkl'))
    with torch.no_grad():
            eids = []
            for batch in test_loader:
                # two graphs are concatenated
                test_edges, test_nodes, eid = genEdgeBatch(g, batch, graph_a, graph_b, adj_a, adj_b, type_a_dict, type_b_dict, num_hops = args.n_layers + 1, num_neighbors = args.num_neighbors)
                #print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
                eids += eid
                emb = model_gan(g, test_edges, test_nodes)
                # emb = g.ndata['features']
                score = model_gan.fc(emb[batch[:, 0]]*emb[batch[:, 1]+ offset]).squeeze() #.sum(dim=1)
                roc_score = roc_auc_score(batch[:,2].numpy(), score.detach().cpu().numpy())
                best_f1 = 0
                for i in range(10):
                    f1 = f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i ) 
                    best_f1 = max(best_f1, f1)
                    #print('ths:{}, f1:{}'.format(i, f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i )))
                # embed()
                print('Test AUC_ROC:{}, Best F1:{}'.format(roc_score, best_f1))

            g.remove_edges(eids)
        
    if args.validation:
        writer.close()
        writer1.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN')
    parser.add_argument("--preprocess", type=bool, default=False,
            help="whether generate new gaph")
    parser.add_argument("--concat", type=bool, default=False,
            help="whether concat at each hidden layer")

    parser.add_argument("--gat", action='store_true',
            help="whether RGCN or RGAT is chosen")
    
    parser.add_argument("--model-opt", type=int, default=1,
            help="[0: triplet loss, 1: binary classification]")

    parser.add_argument("--embedding", type=bool, default=False,
            help="whether h0 is updated")
    parser.add_argument("--validation", type=bool, default=False,
            help="whether draw pr-curve")
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=10,
            help="number of neighbors to be sampled")
    parser.add_argument("--num-negatives", type=int, default=10,
            help="number of negative links to be sampled")
    parser.add_argument("--num-test-negatives", type=int, default=10,
            help="number of negative links to be sampled in test setting")
    parser.add_argument("--n-hidden", type=int, default=50,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--burnin", type=int, default=-1,
            help="when to use hard negatives")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--dump", action='store_true',
            help="dump trained models (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--model-id", type=str,
        help="Identifier of the current model")
    parser.add_argument("--pretrain_path", type=str, default="/shared/data/qiz3/data/enwik9.bin",
        help="pretrained fastText path")
    args = parser.parse_args()


    print(args)

    main(args)

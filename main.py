import module
import utils
import argparse, pickle, time
from IPython import embed
from GraphBuilder import Graph
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


def materializeGraph(dgl_graph, feat_graph, args):
    features = torch.FloatTensor(feat_graph.features)

    norm = 1. / dgl_graph.in_degrees().float().unsqueeze(1)
    node_id = dgl_graph.nodes()
    #edge_type = torch.zeros(dgl_graph.edges()[0].shape, dtype=torch.long)
    edge_type = torch.LongTensor(feat_graph.edge_type)
    num_type = torch.max(edge_type) + 1
    edge_type = torch.cat(( edge_type + 1, edge_type + num_type + 1, 
        torch.zeros(len(feat_graph.id2idx), dtype=torch.long)), 0)
    assert edge_type.shape[0] == dgl_graph.edges()[0].shape[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        norm = norm.cuda()
        node_id = node_id.cuda()
        edge_type = edge_type.cuda()

    
    
    dgl_graph.ndata.update({'id': node_id, 'norm': norm})
    dgl_graph.edata.update({'rel_type': edge_type})
    #embed()
    dgl_graph.ndata['features'] = features
    #dgl_graph.ndata['norm'] = norm

    return dgl_graph, features, num_type

def genEdgeBatch(g, train_data, graph_a, graph_b, adj_a, adj_b, type_a_dict, type_b_dict, add_edge = True, num_hops = 1, num_neighbors = 10):
    train_data = train_data.numpy()
    nodes_a, nodes_b = set(train_data[:, 0].tolist()), set(train_data[:, 1].tolist())
    #type_a_dict, type_b_dict = dict(), dict()
    #adj_a, adj_b = defaultdict(list), defaultdict(list)

    #edge_type_a, edge_type_b = torch.LongTensor(graph_a.edge_type), torch.LongTensor(graph_b.edge_type)
    #num_type_a, num_type_b = torch.max(edge_type_a).item() + 1, torch.max(edge_type_b).item() + 1

    nodes = [list(nodes_a) + list(map(lambda x:x+len(graph_a.id2idx), nodes_b))]

    edge_indices = defaultdict(list)
    eids = []

    left_nodes, right_nodes = set(), set()
    
    for i in range(train_data.shape[0]):
        left_nodes.add(train_data[i, 0])
        right_nodes.add(train_data[i, 1])
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
            e_id = g.edge_id(m+len(graph_a.id2idx), train_data[i, 0])

            #attn_edges.append(-type_b_dict[(m, train_data[i,1])] - 1)
            for sub_edge in type_b_dict[(m, train_data[i,1])]:
                edge_indices[-sub_edge - 1].append(e_id)
            eids.append(e_id)

    if num_hops > 1:
        nodes.append(list(left_nodes) + list(map(lambda x:x+len(graph_a.id2idx), right_nodes)))
        for node_id in list(left_nodes):
            for n in random.sample(adj_a[node_id], min(num_neighbors, len(adj_a[node_id])) ):
                for sub_edge in type_a_dict[(n, node_id)]:
                    edge_indices[sub_edge + 1].append(g.edge_id(n, node_id))
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

    #print(num_type_a, num_type_b)
    assert num_type_a == num_type_b
    
    num_edges = g.number_of_edges()

    g.ndata['features'] = torch.cat([torch.FloatTensor(graph_a.features), torch.FloatTensor(graph_b.features)], 0).cuda()

    #print(g.number_of_edges())
    #g = g.subgraph(nodes)
    #print(g.number_of_edges())

    #print(len(nodes_a), len(nodes_b))
    #nodes_training = []

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

def outputGATNE(graph_a, graph_b, train_data, test_data, out1, out2):
    #g = DGLGraph()
    #g.add_nodes(len(graph_a.id2idx) + len(graph_b.id2idx))
    
    #g.add_edges(graph_a.edge_src, graph_a.edge_dst)
    #g.add_edges(graph_a.edge_dst, graph_a.edge_src)
    for a,b,t in zip(graph_a.edge_src, graph_a.edge_dst, graph_a.edge_type):
        out1.write('{} {} {}\n'.format(t, a, b))
    for a,b,t in zip(graph_b.edge_src, graph_b.edge_dst, graph_b.edge_type):
        out1.write('{} {} {}\n'.format(t, a+len(graph_a.id2idx), b+len(graph_a.id2idx)))

    for i in range(train_data.shape[0]):
        out1.write('{} {} {}\n'.format(0, train_data[i, 0], train_data[i, 1] + len(graph_a.id2idx)))
    
    out2.write('{} {}\n'.format(graph_a.features.shape[0] + graph_b.features.shape[0], graph_a.features.shape[1]))
    for i in range(graph_a.features.shape[0]):
        out2.write('{} {}\n'.format(i, ' '.join(map(str, graph_a.features[i].tolist()))))
    for i in range(graph_b.features.shape[0]):
        out2.write('{} {}\n'.format(i+len(graph_a.id2idx), ' '.join(map(str, graph_b.features[i].tolist()))))
    #g.ndata['features'] = torch.cat([torch.FloatTensor(graph_a.features), torch.FloatTensor(graph_b.features)], 0).cuda()

    #return g

def main(args):
    # load and preprocess dataset
    # data = load_data(args)
    #print('here')
    data_a = 'fb'
    tmp_a = pickle.load(open('data/old/{}_graph.p'.format(data_a), 'rb'))
    data_b = 'imdb'
    tmp_b = pickle.load(open('data/old/{}_graph.p'.format(data_b), 'rb'))

    '''
    dist_a, dist_b = defaultdict(int), defaultdict(int) 
    for t in tmp_a.edge_type:
        dist_a[t] += 1
    for t in tmp_b.edge_type:
        dist_b[t] += 1
    print(dist_a, dist_b)
    return
    '''
    #print('here')
    #neg_head, neg_tail = utils.readFile('./data/positive_labels.txt', './data/negative_labels.txt', includePos = False)
    train_data_person, train_data_film, val_data, test_data = utils.generateTrainWithType('./data/total.txt', tmp_a, tmp_b, 0.1, 0.8)

    #train_data = np.concatenate([train_data_person, train_data_film[:int(train_data_film.shape[0]*0.1), :]],axis = 0)
    #train_data = train_data_person[:int(train_data_person.shape[0]*0.1), :]
    #train_data = np.concatenate([train_data_film, train_data_person[:int(train_data_person.shape[0]*0.1), :]],axis = 0)
    train_data = np.concatenate([train_data_film, train_data_person], axis = 0)
    print(train_data.shape)
    #test_id = np.asarray(utils.genEntAlignDataset(test_data, args.batch_size, len(tmp_a.id2idx), len(tmp_b.id2idx), args.num_test_negatives, combine = False))

    #return

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
    #print('here')
    g, num_rel, offset, adj_a, adj_b, type_a_dict, type_b_dict = genSubGraph(tmp_a, tmp_b, args.n_layers+1)
    #outputGATNE(tmp_a, tmp_b, train_data, train_data, open('data/GATNE_train.txt', 'w'), open('data/GATNE_feature.txt', 'w'))
    #g  = mergeGraph(tmp_a, tmp_b, train_data)
    #print(g.number_of_nodes(), g.number_of_edges())
    #pickle.dump(g, open('data/{}_graph.dgl'.format('imdb-fb'), 'wb'))
    #return
    in_feats = g.ndata['features'].shape[1]

    loss_fcn = module.NCE_HINGE()
    #loss_fcn = module.NCE_HINGE_V2()
    #loss_fcn = module.NCE_HINGE_MOD()
    #loss_fcn = module.NCE_SIGMOID()
    #loss_fcn = nn.BCEWithLogitsLoss()
    #loss_fcn = torch.nn.HingeEmbeddingLoss()
    #test_id = torch.LongTensor(test_id)
    #model = module.BatchPairwiseDistMult(dim=2 * args.n_hidden if args.concat else args.n_hidden)
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


    #loss_fcn = module.NCE_CONTRAST()


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
    #writer.add_graph(model_gan, [edge_indices, torch.LongTensor(train_ids), args.batch_size, args.num_negatives, args.n_hidden, offset])
    for epoch in range(args.n_epochs):
        model_gan.train()
        model.train()
        print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
        train_ids = torch.LongTensor(utils.genEntAlignDataset(train_data, args.batch_size, len(tmp_a.id2idx), len(tmp_b.id2idx), args.num_negatives, combine = False))
        #g_1, edge_indices, training_nodes, eids = genEdgeBatch(g, train_ids, tmp_a, tmp_b, adj_a, adj_b, type_a_dict, type_b_dict)
        training_loss = 0.0
        eids = []
        for batch in tqdm(tdata.DataLoader(train_ids, batch_size=args.batch_size*(args.num_negatives+1), shuffle=False)):
            test_edges, test_nodes, eid = genEdgeBatch(g, batch, tmp_a, tmp_b, adj_a, adj_b, type_a_dict, type_b_dict, num_hops = args.n_layers + 1, num_neighbors = args.num_neighbors)
            #print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
            eids += eid
            emb = model_gan(g, test_edges, test_nodes)
            output_a, output_b = emb[batch[:, 0]].view(-1, args.num_negatives+1, 2 * args.n_hidden), emb[batch[:, 1] + offset].view(-1, args.num_test_negatives+1, 2 * args.n_hidden)
            #g.remove_edges(eid)
            logits = model(output_a, output_b)
            loss = loss_fcn(logits)
            training_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #train_ids = torch.LongTensor(utils.genEntAlignDataset(train_data, args.batch_size, len(tmp_a.id2idx), len(tmp_b.id2idx), args.num_negatives, combine = False))
        #emb = model_gan(g_1, edge_indices, training_nodes)
        print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
        g.remove_edges(eids)
        print(len(eids), len(set(eids)))
        print("Number of deleted edges:{}, Number of nodes:{}, Number of edges:{}".format(len(eids), g.number_of_nodes(), g.number_of_edges()))
        del emb
        torch.cuda.empty_cache()
        #loss.detach()
        
        print('Epoch:{}, loss:{}'.format(epoch, training_loss.detach().item()))

        
        #break 

        if epoch >= 6:
            test_id = torch.LongTensor(pickle.load(open('test_neg_edit_blocking_v2.p', 'rb')))
            
            #print(len(test_nodes))
            #for k in test_edges:
            #    print(k, len(test_edges[k]))
            #emb = model_gan(g, test_edges, test_nodes)

            ## Evaluation Starts
            with torch.no_grad():
                model.eval()
                model_gan.eval()
                count = 0
                test_loss = 0.0
                label = np.zeros((len(test_data) * (2*args.num_test_negatives+1)), dtype=np.int32)
                attn_weight_a, attn_weight_b = np.zeros((len(test_data), args.num_neighbors)), np.zeros((len(test_data), args.num_neighbors))
                idx = 0 
                while idx < len(test_data) * (2*args.num_test_negatives+1):
                    label[idx] = 1
                    idx += 2 * args.num_test_negatives + 1
                pred =np.zeros((len(test_data) * (2*args.num_test_negatives+1)))
                idx = 0
                t1 = time.time()
                line_num = 0 
                for batch in tdata.DataLoader(test_id, batch_size=args.test_batch_size*(args.num_test_negatives+1), shuffle=False):
                    test_edges, test_nodes, __ = genEdgeBatch(g, batch, tmp_a, tmp_b, adj_a, adj_b, type_a_dict, type_b_dict, num_hops = args.n_layers + 1)
                    emb = model_gan(g, test_edges, test_nodes)
                    output_a, output_b = emb[batch[:, 0]].view(-1, args.num_test_negatives+1, 2 * args.n_hidden), emb[batch[:, 1] + offset].view(-1, args.num_test_negatives+1, 2 * args.n_hidden)
                    logits = model(output_a, output_b)
                    pred[idx:idx+logits.shape[0]*logits.shape[1]] = np.reshape(logits.data.cpu().numpy(), -1)
                    idx += logits.shape[0]*logits.shape[1]
                    test_loss += loss_fcn(logits)
                    if count % 10 == 0:
                        print("Progress: {}/{}".format(count, int(test_data.shape[0] / args.test_batch_size)))
                    count += 1
                t2 = time.time()
                pred = (pred.max() - pred) / pred.max()
                print("Inference time is {:.4f}s".format(t2-t1))

                sub_pred = np.zeros((2*args.num_test_negatives+1))
                hit_1 = 0
                idx = 0
                correct_set = set()
                while idx < len(test_data) * (2*args.num_test_negatives+1):
                    sub_pred = pred[idx:idx+(2*args.num_test_negatives+1)]
                    if np.argsort(sub_pred)[::-1].tolist()[0] == 0:
                        hit_1 += 1
                        correct_set.add(idx)
                    idx += 2 * args.num_test_negatives + 1
                print("Epoch:{}, training loss:{}, validation loss:{}, hit@1:{}".format(epoch, training_loss, test_loss, hit_1 / len(test_data)))
                person_test = pickle.load(open('test_neg_edit_blocking_person_id.p', 'rb'))
                film_test = pickle.load(open('test_neg_edit_blocking_film_id.p', 'rb'))
                pickle.dump(correct_set, open('results/{}_epoch_{}_correct.p'.format(args.model_id, epoch), 'wb'))
                pickle.dump({'pred':pred, 'label':label}, open('results/{}_epoch_{}_pr.p'.format(args.model_id, epoch), 'wb'))
                #pickle.dump([attn_weight_a, attn_weight_b], open('results/{}_epoch_{}_attention_weights.p'.format(args.model_id, epoch), 'wb'))
                if args.validation:
                    writer.add_pr_curve('pr_curve_validation_sub_type', label[person_test], pred[person_test], epoch)
                    writer1.add_pr_curve('pr_curve_validation_sub_type', label[film_test], pred[film_test], epoch)
        
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
    
    parser.add_argument("--model-opt", type=int, default=-1,
            help="corresponded model id in the readme")

    parser.add_argument("--embedding", type=bool, default=False,
            help="whether h0 is updated")
    parser.add_argument("--validation", type=bool, default=False,
            help="whether draw pr-curve")
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=3e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=10,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=36,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=10,
            help="number of neighbors to be sampled")
    parser.add_argument("--num-negatives", type=int, default=10,
            help="number of negative links to be sampled")
    parser.add_argument("--num-test-negatives", type=int, default=10,
            help="number of negative links to be sampled in test setting")
    parser.add_argument("--n-hidden", type=int, default=64,
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
    args = parser.parse_args()


    print(args)

    main(args)
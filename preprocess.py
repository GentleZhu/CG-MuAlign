import sys,pickle
from IPython import embed
from collections import defaultdict
import matplotlib.pyplot as plt
#from FeaturePipeline import FeatureGenerator, warp_feature
import numpy as np
from tqdm import tqdm
import fasttext as ft
import torch
#from nltk.corpus import wordnet
import csv

class FeatureGenerator(object):
	"""Name Embedding FeatureGenerator"""
	def __init__(self, model_path):
		super(FeatureGenerator, self).__init__()
		self.model = ft.load_model(model_path)

	def generateEmbFeature(self, name, sent=True):
		if sent == True:       
			return self.model.get_sentence_vector(name.replace('"',''))
		else:     
			return self.model.get_word_vector(name.replace('"',''))

# the interface to generate a numpy feature matrix for every node in the graph
def warp_feature(model, source, mapping):
	feature_matrix = np.zeros((len(source), 100))

	row = 0
	none_cnt = 0
	for k in source:
		# only for amc-wiki dataset, uncomment two following two lines
		#if source[k]['attr'][0]:
		for attr in source[k][1]:
			#embed()
			feature_matrix[mapping[k], :] = model.generateEmbFeature(attr, sent=True)
		#if source[k]['attr'][0] is not None:
		#	feature_matrix[row, :] = model.generateEmbFeature(source[k]['attr'][0], sent=True)
		#else:
		#	none_cnt += 1
	return feature_matrix


def generateTrainWithType(in_path, graph_a, graph_b):
	train_data, val_data, test_data = [], [], [] 
	with open(in_path+'train.csv') as IN:
		IN.readline()
		left_set, right_set = set(), set()
		for line in IN:
			tmp = line.strip().split(',')
			if tmp[0] not in left_set and tmp[1] not in right_set:
				left_set.add(tmp[0])
				right_set.add(tmp[1])
			else:
				continue
			#print(graph_a.entity_table['ID_{}'.format(tmp[0])])
			#print(graph_b.entity_table['ID_{}'.format(tmp[1])])
			#print(tmp[2])
			#embed()
			train_data.append([graph_a.id2idx['ID_{}'.format(tmp[0])], 
				graph_b.id2idx['ID_{}'.format(tmp[1])], int(tmp[2])])
	# embed()
	with open(in_path+'valid.csv') as IN:
		IN.readline()
		left_set, right_set = set(), set()
		for line in IN:
			tmp = line.strip().split(',')
			if tmp[0] not in left_set and tmp[1] not in right_set:
				left_set.add(tmp[0])
				right_set.add(tmp[1])
			else:
				continue
			#print(graph_a.entity_table['ID_{}'.format(tmp[0])])
			#print(graph_b.entity_table['ID_{}'.format(tmp[1])])
			#print(tmp[2])
			#embed()
			val_data.append([graph_a.id2idx['ID_{}'.format(tmp[0])], 
				graph_b.id2idx['ID_{}'.format(tmp[1])], int(tmp[2])])
	with open(in_path+'test.csv') as IN:
		IN.readline()
		left_set, right_set = set(), set()
		for line in IN:
			tmp = line.strip().split(',')
			if tmp[0] not in left_set and tmp[1] not in right_set:
				left_set.add(tmp[0])
				right_set.add(tmp[1])
			else:
				continue
			#print(graph_a.entity_table['ID_{}'.format(tmp[0])])
			#print(graph_b.entity_table['ID_{}'.format(tmp[1])])
			#print(tmp[2])
			#embed()
			test_data.append([graph_a.id2idx['ID_{}'.format(tmp[0])], 
				graph_b.id2idx['ID_{}'.format(tmp[1])], int(tmp[2])])

	return torch.LongTensor(train_data), torch.LongTensor(val_data), torch.LongTensor(test_data)


class Graph(object):
	"""docstring for Graph"""
	def __init__(self):
		super(Graph, self).__init__()
		#self.relation_list = relation_list
		self.id2idx = {}
		self.entity_table = {}
		self.features = None
		self.edge_src = []
		self.edge_dst = []
		self.edge_type = []
	
	def buildGraph(self, table):
		# self.self.entity_table_table = self.entity_table_path
		#
		with open(table) as IN:
			spamreader = csv.reader(IN, delimiter=',')
			# embed()
			# fields = IN.readline().strip().split(',')
			fields = next(spamreader)
			# self.entity_table, id2idx = {}, {}
			type_list, type_dict = [], {}
			attr_list = []
			for idx,field in enumerate(fields[1:]):
				if '_' in field:
					type_list.append(field.split('_')[0])
				else:
					attr_list.append(field)
			print(type_list)
			edge_list = []
			for line in spamreader:
				# print(line)
				tmp = line
				for idx, value in enumerate(tmp[1:]):
					if idx < len(type_list):
						if idx == 0:
							_ID = 'ID_{}'.format(tmp[0])
							self.entity_table[_ID] = [type_list[idx], value]
							self.id2idx[_ID] = len(self.id2idx)
							target_id = self.id2idx[_ID]
						else:
							_id = '{}_{}'.format(type_list[idx],value)
							if _id not in self.entity_table:
								self.entity_table[_id] = [type_list[idx], value]
								#_ID = '{}_{}'.format(tm, type_list[idx])
								self.id2idx[_id] = len(self.id2idx)
							#edge_list.append([target_id, idx, id2idx[value]])
							self.edge_src.append(target_id)
							self.edge_dst.append(self.id2idx[_id])
							self.edge_type.append(idx - 1)
					else:
						self.entity_table[_ID].append(value)

			feat = FeatureGenerator('/shared/data/qiz3/data/enwik9.bin')
			#for tmp in triples
				#if tmp[0] in self.id2idx and tmp[1] in self.id2idx and tmp[2] in self.relation_list:
				#if tmp[0] in self.id2idx and tmp[2] in self.id2idx:
					#g.add_edges()
					#self.edge_list.append([self.id2idx[tmp[0]], self.id2idx[tmp[2]], self.relation_list.index(tmp[1])])
			# embed()	
			self.features = warp_feature(feat, self.entity_table, self.id2idx)
		#assert self.features.shape()[0] == len(self.id2idx)
						
				

def checkTest(mapping_a, mapping_b, in_file):
	type_cnt_a, type_cnt_b = defaultdict(int), defaultdict(int)
	str_pair = set()
	with open(in_file) as IN:
		for line in IN:
			tmp = line.strip().split('\t')
			if tmp[0] in mapping_a and tmp[1] in mapping_b:
				str_pair.add('{}_{}'.format(tmp[0], tmp[1]))
				for x in mapping_a[tmp[0]]['type']:
					type_cnt_a[x]+= 1
				for x in mapping_b[tmp[1]]['type']:
					type_cnt_b[x]+= 1
	print("Len of original data is {}".format(len(str_pair)))
	print(type_cnt_a, type_cnt_b)

if __name__ == '__main__':
	dataset = 'itunes' #imdb
	graph_a, graph_b = Graph(), Graph()
	graph_a.buildGraph('data/itunes_amazon_exp_data/exp_data/tableA.csv')
	graph_b.buildGraph('data/itunes_amazon_exp_data/exp_data/tableB.csv')
	
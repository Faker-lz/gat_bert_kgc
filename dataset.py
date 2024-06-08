'''
Author: WLZ
Date: 2024-06-03 21:05:26
Description: 
'''
import os 
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class KnowledgeGraphDataset(Dataset):
    def __init__(self, graph_part_dir, entity2id, relation2id):
            self.graph_part_dir = graph_part_dir
            self.graph_part_files = [os.path.join(graph_part_dir, f) for f in os.listdir(graph_part_dir) if f.endswith('.txt')]
            self.entity2id = entity2id
            self.relation2id = relation2id

    def __len__(self):
        return len(self.graph_part_files)

    def __getitem__(self, idx):
        part_file = self.graph_part_files[idx]
        triples = load_data(part_file, entity2id=self.entity2id, relation2id=self.relation2id)
        adj_matrix = build_adjacency_matrix(triples, self.entity2id, True)
        return adj_matrix, triples

def load_data(filepath, load_all=False, entity2id=None, relation2id=None):
    entity_set = set()
    relation_set = set()
    triples = []

    if load_all:
        with open(filepath, 'r') as file:
            for line in file:
                head, relation, tail = line.strip().split('\t')
                entity_set.add(head)
                entity_set.add(tail)
                relation_set.add(relation)
                triples.append((head, relation, tail))

        entity2id = {entity: idx for idx, entity in enumerate(entity_set)}
        relation2id = {relation: idx for idx, relation in enumerate(relation_set)}

        return triples, entity2id, relation2id
    else:
        with open(filepath, 'r') as file:
            for line in file:
                head, relation, tail = line.strip().split('\t')
                triples.append((entity2id[head], relation2id[relation], entity2id[tail]))
        return triples

def build_edge_index(triples, entity2id, is_id=False):
    """
    构建PyTorch Geometric格式的边索引
    :param triples: 知识图谱三元组列表，每个三元组是(head, relation, tail)
    :param entity2id: 实体到ID的映射字典
    :return: PyTorch Geometric格式的边索引
    """
    edges = []
    for head, _, tail in triples:
        if not is_id:
            head_id = entity2id[head]
            tail_id = entity2id[tail]
        else:
            head_id = head
            tail_id = tail
        edges.append([head_id, tail_id])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def build_adjacency_matrix(triples, entity2id, is_id=False):
    """
    构建邻接矩阵
    :param triples: 知识图谱三元组列表，每个三元组是(head, relation, tail)
    :param entity2id: 实体到ID的映射字典
    :return: 邻接矩阵
    """
    n_entities = len(entity2id)
    adj_matrix = torch.zeros((n_entities, n_entities), dtype=torch.float32)
    for head, _, tail in triples:
        if is_id:
            adj_matrix[head, tail] = 1.0
        else:
            head_id = entity2id[head]
            tail_id = entity2id[tail]
            adj_matrix[head_id, tail_id] = 1.0
    return adj_matrix

if __name__ == '__main__':
    triples, entity2id, relation2id = load_data(os.path.join(os.path.dirname(__file__) ,r'data\WN18RR\train.txt'))
    adj_matrix = build_edge_index(triples, entity2id, relation2id)
    dataset = KnowledgeGraphDataset(triples, entity2id, relation2id)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in dataloader:
        print(data)

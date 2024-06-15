'''
Author: WLZ
Date: 2024-06-03 21:05:26
Description: 
'''
import os 
from link_graph import LinkGraph
from torch.utils.data import Dataset, DataLoader
from utils import build_adjacency_matrix, build_edge_index


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
        adj_matrix = build_adjacency_matrix(triples, self.entity2id)
        return adj_matrix, triples

class KnowledgeGtaphTestDataset(Dataset):
    def __init__(self, test_data_path, entity2id, relation2id) -> None:
        super().__init__()
        self.triples = load_data(test_data_path, entity2id=entity2id, relation2id=relation2id)
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.link_graph = LinkGraph(self.triples ,entity2id)
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, index):
        return self.triples[index]
    
class TailEntityDataset(Dataset):
    def __init__(self, entities, link_graph):
        self.entities = entities
        self.link_graph = link_graph

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, idx):
        tail_entity = self.entities[idx]
        tail_neighbors = self.link_graph.get_neighbors(tail_entity)

        # TODO实现或复用获取adj的方法
        adj_matrix = self.link_graph.get_node_adj(tail_neighbors)

        return tail_entity, adj_matrix



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

if __name__ == '__main__':
    triples, entity2id, relation2id = load_data(os.path.join(os.path.dirname(__file__) ,r'data\WN18RR\train.txt'))
    adj_matrix = build_edge_index(triples, entity2id, relation2id)
    dataset = KnowledgeGraphDataset(triples, entity2id, relation2id)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in dataloader:
        print(data)


import torch
from collections import deque
from logger_config import logger
from utils import build_adjacency_matrix, build_sparse_adjacency_matrix


class LinkGraph():
    def __init__(self, triples,  entities2id):
        self.graph = self.generate_direct_graph(triples)
        self.entities2id = entities2id
        self.triples = triples

    def generate_direct_graph(self, triples):
        graph = dict()
        for head_id, _, tail_id in triples:
            if tail_id not in graph.keys():
                graph[tail_id] = list()

            graph[tail_id].append(head_id)
        
        logger.info('Done build link graph with {} nodes'.format(len(graph)))
        return graph

    def get_neighbors(self, node_id, k_step=3):
        neighbors = set()
        neighbors.add(node_id)

        queue = deque()
        if k_step <= 0:
            return neighbors
        
        queue.append(node_id)
        for _ in range(k_step + 1):
            len_que = len(queue)
            for _ in range(len_que):
                node = queue.popleft()
                for neig in self.graph.get(node, set()):
                    if neig not in neighbors:
                        neighbors.add(neig)
                        queue.append(neig)
        return neighbors
    
    def get_node_adj(self, nodes):
        entities_num = len(self.entities2id)
        adj_matrix = torch.eye(entities_num)
        select_triples = list()
        for head, relation, tail in self.triples:
            if head in nodes and tail in nodes:
                # 既选择了头尾实都命中的三元组，又选择了头或者尾单一命中的三元组，丰富邻接矩阵。
                select_triples.append((head, relation, tail))
        # logger.info(f'Get Node num: {len(nodes)}; get triples num: {len(select_triples)}')
        # adj_matrix = build_adjacency_matrix(select_triples, self.entities2id)
        adj_matrix = build_sparse_adjacency_matrix(select_triples, self.entities2id)
        return adj_matrix



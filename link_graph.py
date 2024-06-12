
from collections import deque
from logger_config import logger


class LinkGraph():
    def __init__(self,triples):
        self.graph = self.generate_direct_graph(triples)

    def generate_direct_graph(self, triples):
        graph = dict()
        for head_id, _, tail_id in triples:
            if tail_id not in graph.keys():
                graph[tail_id] = list()

            graph[tail_id].append(head_id)
        
        logger.info('Done build link graph with {} nodes'.format(len(graph)))
        return graph

    def get_neighbors(self, node_id, k_step=3):
        neighbors = set(node_id)
        queue = deque()
        if k_step <= 0:
            return neighbors
        
        queue.append([node_id])
        for _ in range(k_step):
            len_que = len(queue)
            for _ in range(len_que):
                node = queue.popleft()
                for neig in self.graph.get(node, set()):
                    if neig not in neighbors:
                        neighbors.add(neig)
                        queue.append(neig)
        
        return neighbors



import os 
import torch
from collections import OrderedDict

from torch.utils.data import DataLoader
from knowledge_graph_gat import KnowledgeGraphGAT
from dataset import KnowledgeGtaphTestDataset, TailEntityDataset


class Predictor:
    def __init__(self) -> None:
        self.model = None
        self.train_args = dict()

    def load(self, model_path, all_entity2id, all_relation2id):
        assert os.path.exists(model_path)
        ckt_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        # build_tokenizer(self.train_args)
        self.model = KnowledgeGraphGAT(3, len(all_entity2id), len(all_relation2id), 768, 32, 128, 768, 0.2, 0.2, 0.05, 3)

        # DataParallel will introduce 'module.' prefix
        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def get_hr_embeddings(self, all_entities2id, batch_size, test_data_path):
        hr_emb_list, tail_emb_list, target_list = list(), list()

        test_dataset = KnowledgeGtaphTestDataset(test_data_path)
        test_dataloader = DataLoader(test_dataset, batch_size)
        for triples in test_dataloader:
            head_id, relation_id, tail_id = zip(*triples)

            nodes = [test_dataset.link_graph.get_neighbors(node) for node in head_id]
            nodes = set.union(*nodes)
            nodes = list(nodes)

            adj = test_dataset.get_test_nodes_adj(nodes)

            hr_emb_list.append(self.model.compute_embedding(head_id, relation_id, tail_id, adj, nodes))

            target_list.append(tail_id)

        # 根据all_entities2id 按照batch_size获取每个tail实体经过GAT融合的embedding，其中也是选取具体的tail的3跳邻居节点作为构造邻接矩阵的依据
        tail_entities = list(all_entities2id.values())
        tail_dataset = TailEntityDataset(tail_entities, test_dataset.link_graph)
        tail_dataloader = DataLoader(tail_dataset, batch_size)

        tail_emb_list = []
        # 批量处理尾部实体的嵌入计算
        for batch in tail_dataloader:
            tail_entity_batch, adj_batch = batch

            tail_emb_batch = self.model.compute_embedding(tail_entity_batch, [relation_id[0]] * len(tail_entity_batch), [tail_id[0]] * len(tail_entity_batch), adj_batch, tail_entity_batch)
            tail_emb_list.append(tail_emb_batch)

        return torch.cat(hr_emb_list, dim=0), torch.cat(tail_emb_list, dim=0), target_list
    
    @torch.no_grad()
    def get_tail_embedding(self):
        # 获取所有实体GAT融合后的embedding
        pass
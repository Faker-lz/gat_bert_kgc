import os 
import tqdm
import torch
from collections import OrderedDict

from logger_config import logger
from torch.utils.data import DataLoader
from knowledge_graph_gat import KnowledgeGraphGAT
from dataset import KnowledgeGtaphTestDataset, TailEntityDataset


class Predictor:
    def __init__(self, device) -> None:
        self.model = None
        self.train_args = dict()
        self.device = device

    def load(self, model_path, all_entity2id, all_relation2id):
        assert os.path.exists(model_path)
        ckt_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        # self.train_args.__dict__ = ckt_dict['args']
        # self._setup_args()
        # build_tokenizer(self.train_args)
        self.model = KnowledgeGraphGAT(2, len(all_entity2id), len(all_relation2id), 768, 32, 128, 768, 0.2, 0.2, 0.05, 3)
        self.model = self.model.to(self.device)

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
    def get_hr_embeddings(self, all_entities2id, all_relation2id, batch_size, test_data_path, train_data_path,device):
        hr_emb_list, tail_emb_list, target_list = list(), list(), list()

        test_dataset = KnowledgeGtaphTestDataset(test_data_path, train_data_path, all_entities2id, all_relation2id)
        test_dataloader = DataLoader(test_dataset, batch_size)

        logger.info('compute hr embedding ...')
        for triples in tqdm.tqdm(test_dataloader):
            head_id, relation_id, tail_id = triples

            nodes = [test_dataset.link_graph.get_neighbors(node.item()) for node in head_id]
            nodes = set.union(*nodes)

            adj = test_dataset.link_graph.get_node_adj(nodes)
            adj = adj.to(self.device)
            nodes = torch.LongTensor(list(nodes)).to(self.device)

            hr_emb_list.append(self.model.compute_embedding(head_id, relation_id, None, adj, task='eval_hr',nodes_id=nodes))

            target_list.append(tail_id)

        # 根据all_entities2id 按照batch_size获取每个tail实体经过GAT融合的embedding，其中也是选取具体的tail的n跳邻居节点作为构造邻接矩阵的依据
        tail_entities = list(all_entities2id.values())
        tail_dataset = TailEntityDataset(tail_entities, test_dataset.link_graph)
        tail_dataloader = DataLoader(tail_dataset, batch_size=1024)

        logger.info('compute tail embedding ...')
        tail_emb_list = []
        # 批量处理尾部实体的嵌入计算
        for batch_tails in tqdm.tqdm(tail_dataloader):
            nodes = [tail_dataset.link_graph.get_neighbors(tail.item()) for tail in batch_tails]
            nodes = set.union(*nodes)
            nodes = torch.LongTensor(list(nodes)).to(self.device)

            adj = tail_dataset.link_graph.get_node_adj(nodes).to(self.device)

            tail_emb_batch = self.model.compute_embedding(None, None, batch_tails, adj,  'eval_tail', nodes)
            tail_emb_list.append(tail_emb_batch)
        return torch.cat(hr_emb_list, dim=0).to(device), torch.cat(tail_emb_list, dim=0).to(device), torch.cat(target_list, dim=0).to(device)

        # tail_emb = self.model.entity_embeddings.weight
        # tail_emb = torch.nn.functional.normalize(tail_emb, -1)

        # return torch.cat(hr_emb_list, dim=0).to(device), tail_emb.to(device), torch.cat(target_list, dim=0).to(device)



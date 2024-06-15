'''
Author: WLZ
Date: 2024-06-03 20:05:46
Description: 
'''
import torch
import torch.nn as nn
from graph_attention_network import MultiLayerGAT
import torch.nn.functional as F

class KnowledgeGraphGAT(nn.Module):
    def __init__(self, n_layers, n_entities, n_relations, entity_dim, relation_dim, hid_dim, out_dim, dropout, alpha, temperature, nheads,device='cuda'):
        super(KnowledgeGraphGAT, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_layers = n_layers
        self.device = device
        self.log_inv_t = nn.Parameter(torch.tensor(1.0 / temperature).log())

        
        self.entity_embeddings = nn.Embedding(n_entities, entity_dim).to(device)
        self.relation_embeddings = nn.Embedding(n_relations, relation_dim).to(device)
        
        self.gat = MultiLayerGAT(nlayer=n_layers, nfeat=entity_dim, nhid=hid_dim, 
                                 noutfeat=out_dim, dropout=dropout, alpha=alpha, nheads=nheads).to(device)
        
        self.fusion_linear = nn.Linear(entity_dim + relation_dim, entity_dim).to(device)
        
    def forward(self, head_id, relation_id, tail_id, adj):
        hr_emb, tail_emb = self.compute_embedding(head_id, relation_id, tail_id, adj)
        triples_number = hr_emb.shape[0]

        logits = hr_emb @ tail_emb.T
        logits *= self.log_inv_t.exp()
        target = torch.arange(triples_number).to(self.device)
        return logits, target

    def compute_embedding(self, head_id, relation_id, tail_id, adj, task='training', nodes_id=None):
        head_id = torch.tensor(head_id, dtype=torch.long).to(self.device)
        relation_id = torch.tensor(relation_id, dtype=torch.long).to(self.device)

        if task == 'training':
            tail_id = torch.tensor(tail_id, dtype=torch.long).to(self.device)
            unique_entities, inverse_indices = torch.unique(torch.cat([head_id, tail_id]), return_inverse=True)

            x = self.entity_embeddings(unique_entities).to(self.device)
            sub_adj_matrix = adj[unique_entities][:, unique_entities]
            output = self.gat(x, sub_adj_matrix)

            head_emb = output[inverse_indices[:len(head_id)]]
            relation_emb = self.relation_embeddings(relation_id)
            hr_emb = self.fusion_linear(torch.cat([head_emb, relation_emb], dim=1))
            hr_emb = nn.functional.normalize(hr_emb, dim=1)

            tail_emb = output[inverse_indices[len(head_id):]]
            tail_emb = nn.functional.normalize(tail_emb, dim=1)
            return hr_emb, tail_emb
        else:
            unique_entities, inverse_indices = torch.unique(nodes_id, return_inverse=True)

            x = self.entity_embeddings(unique_entities).to(self.device)
            sub_adj_matrix = adj[unique_entities][:, unique_entities]
            output = self.gat(x, sub_adj_matrix)

            head_indices = torch.hstack([torch.where(unique_entities == h)[0] for h in head_id])
            head_emb = output[head_indices]

            head_emb = output[inverse_indices[:len(head_id)]]
            relation_emb = self.relation_embeddings(relation_id)
            hr_emb = self.fusion_linear(torch.cat([head_emb, relation_emb], dim=1))
            hr_emb = nn.functional.normalize(hr_emb, dim=1)
            return hr_emb


if __name__ == '__main__':
    n_entities = 1000       # 假设有1000个实体
    n_relations = 100       # 假设有100个关系
    entity_dim = 768        # 实体嵌入维度
    relation_dim = 768      # 关系嵌入维度
    dropout = 0.2
    alpha = 0.2
    nheads = 3

    model = KnowledgeGraphGAT(n_entities, n_relations, entity_dim, relation_dim, dropout, alpha, nheads)

    # 示例输入
    head = torch.arange(1000)
    relation = torch.arange(10).repeat_interleave(100)
    tail = torch.arange(1000)
    adj = torch.eye(n_entities)

    x, head_relation_combined = model(head, relation, adj)
    similarity = model.compute_similarity(head_relation_combined, tail)

    print(f"Head and Relation Combined: {head_relation_combined}")
    print(f"Similarity: {similarity}")

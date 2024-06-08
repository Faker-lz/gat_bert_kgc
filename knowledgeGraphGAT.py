'''
Author: WLZ
Date: 2024-06-03 20:05:46
Description: 
'''
import torch
import torch.nn as nn
from graphAttentionNetwork import MultiLayerGAT
import torch.nn.functional as F

class KnowledgeGraphGAT(nn.Module):
    def __init__(self, n_entities, n_relations, entity_dim, relation_dim, dropout, alpha, nheads,device='cuda'):
        super(KnowledgeGraphGAT, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.device = device
        
        self.entity_embeddings = nn.Embedding(n_entities, entity_dim).to(device)
        self.relation_embeddings = nn.Embedding(n_relations, relation_dim).to(device)
        
        self.gat = MultiLayerGAT(nlayer=10, nfeat=entity_dim, nhid=entity_dim, 
                                 noutfeat=entity_dim, dropout=dropout, alpha=alpha, nheads=nheads).to(device)
        
    def forward(self, head, relation, adj):
        head = head.to(self.device)
        relation = relation.to(self.device)
        adj = adj.to(self.device)

        head_emb = self.entity_embeddings(head)  # 头实体嵌入
        relation_emb = self.relation_embeddings(relation)  # 关系嵌入
        
        x = self.entity_embeddings.weight.to(self.device)  # 获取所有实体嵌入作为GAT输入
        x = self.gat(x, adj)  # GAT输出更新后的实体嵌入
        
        head_relation_combined = head_emb + relation_emb  # 头实体和关系嵌入融合（可以用其他方式如拼接）
        
        return x, head_relation_combined

    def compute_similarity(self, head_relation_combined, tail):
        tail = tail.to(self.device)
        tail_emb = self.entity_embeddings(tail)  # 尾实体嵌入
        similarity = F.cosine_similarity(head_relation_combined, tail_emb)  # 计算相似度（这里使用余弦相似度）
        return similarity

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

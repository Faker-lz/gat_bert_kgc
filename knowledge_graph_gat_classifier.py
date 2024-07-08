'''
Author: WLZ
Date: 2024-07-01 21:21:21
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta
from graph_attention_network import MultiLayerGAT

class KnowledgeGraphGATClassifier(nn.Module):
    def __init__(self, n_layers, n_entities, n_relations, entity_dim, relation_dim, hid_dim, out_dim, class_num, dropout, alpha, temperature, nheads,device='cuda'):
        super(KnowledgeGraphGATClassifier, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.class_num = class_num
        self.n_layers = n_layers
        self.device = device
        self.log_inv_t = nn.Parameter(torch.tensor(1.0 / temperature).log())

        self.classes_matrix = init_beta_distribution(n_relations, class_num)
        
        self.entity_embeddings = nn.Embedding(n_entities, entity_dim).to(device)
        
        self.gat = MultiLayerGAT(nlayer=n_layers, nfeat=entity_dim, nhid=hid_dim, 
                                 noutfeat=out_dim, dropout=dropout, alpha=alpha, nheads=nheads).to(device)
        
        self.entities_classifier = nn.Linear(self.out_dim, self.class_num).to(device)
        
    def forward(self, head_id, relation_id, tail_id, nodes, adj):
        head_emb, tail_emb = self.compute_embedding(head_id, relation_id, tail_id, adj, nodes_id=nodes)

        head_class = self.entities_classifier(head_emb)
        tail_class = self.entities_classifier(tail_emb)

        head_one_hot = F.one_hot(head_class.argmax(dim=-1), num_classes=self.class_num).float()
        tail_one_hot = F.one_hot(tail_class.argmax(dim=-1), num_classes=self.class_num).float()
        
        # Prepare the relation matrices for each example in the batch
        relation_id = torch.cat(relation_id)
        relation_matrices = self.classes_matrix[relation_id]
        
        # Compute logits using batched matrix operations
        # Step 1: Batched matrix-vector product for head entities
        head_transformed = torch.bmm(head_one_hot.unsqueeze(1), relation_matrices).squeeze(1)
        
        # Step 2: Batch matrix multiplication with tail one-hot encodings
        logits = torch.mm(head_transformed, tail_one_hot.t())
        
        # The target index (correct tail for each head)
        target = torch.arange(logits.shape[0]).to(self.device)
        
        return logits, target

    def compute_embedding(self, head_id, relation_id, tail_id, adj, task='training', nodes_id=None):
        adj = adj.to_dense()
        if task == 'training':
            head_id = torch.tensor(head_id, dtype=torch.long).to(self.device)
            relation_id = torch.tensor(relation_id, dtype=torch.long).to(self.device)
            tail_id = torch.tensor(tail_id, dtype=torch.long).to(self.device)
            
            unique_entities, inverse_indices = torch.unique(nodes_id, return_inverse=True)

            x = self.entity_embeddings(unique_entities).to(self.device)
            adj = adj[unique_entities][:, unique_entities]
            output = self.gat(x, adj)

            head_indices = torch.hstack([torch.where(unique_entities == h)[0] for h in head_id])
            head_emb = output[head_indices]
            head_emb = nn.functional.normalize(head_emb)

            tail_indices = torch.hstack([torch.where(unique_entities == t)[0] for t in tail_id])
            tail_emb = output[tail_indices]
            tail_emb = nn.functional.normalize(tail_emb)

            return head_emb, tail_emb
        elif task == 'eval_hr':
            head_id = torch.tensor(head_id, dtype=torch.long).to(self.device)
            relation_id = torch.tensor(relation_id, dtype=torch.long).to(self.device)

            unique_entities, inverse_indices = torch.unique(nodes_id, return_inverse=True)

            x = self.entity_embeddings(unique_entities).to(self.device)
            adj = adj[unique_entities][:, unique_entities]
            output = self.gat(x, adj)

            head_indices = torch.hstack([torch.where(unique_entities == h)[0] for h in head_id])
            head_emb = output[head_indices]

            relation_emb = self.relation_embeddings(relation_id)
            hr_emb = self.fusion_linear(torch.cat([head_emb, relation_emb], dim=1))
            hr_emb = nn.functional.normalize(hr_emb, dim=1)
            return hr_emb
        else:
            unique_entities, inverse_indices = torch.unique(nodes_id, return_inverse=True)

            x = self.entity_embeddings(unique_entities).to(self.device)
            adj = adj[unique_entities][:, unique_entities]
            output = self.gat(x, adj)

            tail_indices = torch.hstack([torch.where(unique_entities == t)[0] for t in tail_id])
            tail_emb = nn.functional.normalize(output[tail_indices], dim=1)
            return tail_emb
        
def init_beta_distribution(n_relations, class_num, alpha=2.0, beta=5.0):
    """
    Initialize a tensor using a Beta distribution.

    Parameters:
    - n_relations (int): Number of relations.
    - class_num (int): Number of classes.
    - alpha (float): Alpha parameter of the Beta distribution.
    - beta (float): Beta parameter of the Beta distribution.

    Returns:
    - torch.nn.Parameter: Initialized tensor with values drawn from the Beta distribution.
    """
    # Create a Beta distribution object
    beta_dist = Beta(torch.full((n_relations, class_num, class_num), alpha),
                    torch.full((n_relations, class_num, class_num), beta))
    
    # Sample from the Beta distribution
    return nn.Parameter(beta_dist.sample())

# Example of usage
n_relations = 3
class_num = 4
alpha = 2.0
beta = 5.0
class_matrix = init_beta_distribution(n_relations, class_num, alpha, beta)
class_matrix  # Display the initialized class matrix


if __name__ == '__main__':
    n_entities = 1000       # 假设有1000个实体
    n_relations = 100       # 假设有100个关系
    entity_dim = 768        # 实体嵌入维度
    relation_dim = 768      # 关系嵌入维度
    dropout = 0.2
    alpha = 0.2
    nheads = 3

    model = KnowledgeGraphGATClassifier(n_entities, n_relations, entity_dim, relation_dim, dropout, alpha, nheads)

    # 示例输入
    head = torch.arange(1000)
    relation = torch.arange(10).repeat_interleave(100)
    tail = torch.arange(1000)
    adj = torch.eye(n_entities)

    x, head_relation_combined = model(head, relation, adj)
    similarity = model.compute_similarity(head_relation_combined, tail)

    print(f"Head and Relation Combined: {head_relation_combined}")
    print(f"Similarity: {similarity}")

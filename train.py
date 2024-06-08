'''
Author: WLZ
Date: 2024-06-03 21:11:37
Description: 
'''
import torch
import torch.nn as nn
from torch import optim
from metric import compute_accuracy
from torch.utils.data import DataLoader
from graphAttentionNetwork import MultiLayerGAT
from dataset import KnowledgeGraphDataset, load_data
from utils import move_to_cuda, save_checkpoint, delete_old_ckt


class KnowledgeGraphTrainer:
    def __init__(self, all_file_path, train_file_path, valid_file_path, layers, entity_dim, relation_dim, dropout, 
                 alpha, nheads, batch_size=1, lr=0.01, num_epochs=10, device='cpu'):
        self.all_file_path = all_file_path
        self.train_file_path = train_file_path
        self.valid_file_path = valid_file_path
        self.layers = layers
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs    
        self.device = device
        self.best_metrics = None

        _, self.all_entity2id, self.all_relation2id = load_data(all_file_path, True)
        self.train_dataset = KnowledgeGraphDataset(train_file_path, self.all_entity2id, self.all_relation2id)
        self.valid_dataset = KnowledgeGraphDataset(valid_file_path, self.all_entity2id, self.all_relation2id)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True)
        
        self.n_entities = len(self.all_entity2id)
        self.n_relations = len(self.all_relation2id)
        
        self.model = MultiLayerGAT(layers, entity_dim, entity_dim, entity_dim, dropout, alpha, nheads).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss().to(device)
        
        self.entity_embeddings = nn.Embedding(self.n_entities, entity_dim).to(device)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        self.relation_embeddings = nn.Embedding(self.n_relations, relation_dim).to(device)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        self.fusion_linear = nn.Linear(entity_dim + relation_dim, entity_dim)
        
    def train_epoch(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_index, (adj_matrix, triples) in enumerate(self.train_dataloader):
                if self.device == 'cuda':
                    move_to_cuda(adj_matrix)
                    move_to_cuda(triples)
                
                adj_matrix = adj_matrix.squeeze().to(self.device)
                head, relation, tail = zip(*triples)
                head = torch.tensor(head, dtype=torch.long).to(self.device)
                relation = torch.tensor(relation, dtype=torch.long).to(self.device)
                tail = torch.tensor(tail, dtype=torch.long).to(self.device)
                triples_number = head.shape[0]

                # 获取唯一节点并映射回原始节点索引
                unique_entities, inverse_indices = torch.unique(torch.cat([head, tail]), return_inverse=True)
                x = self.entity_embeddings(unique_entities).to(self.device)
                
                # 构建子图的邻接矩阵
                sub_adj_matrix = adj_matrix[unique_entities][:, unique_entities]

                output = self.model(x, sub_adj_matrix)
                
                head_emb = output[inverse_indices[:len(head)]]
                relation_emb = self.relation_embeddings(relation)
                tail_emb = output[inverse_indices[len(head):]]

                hr = self.fusion_linear(torch.cat([head_emb, relation_emb], dim=1))

                hr = nn.functional.normalize(hr, dim=1)
                tail_emb = nn.functional.normalize(tail_emb, dim=1)

                logits = hr @ tail_emb.T
                target = torch.arange(triples_number).to(self.device)

                loss = self.criterion(logits, target)
                acc1, acc3 , acc10 = compute_accuracy(logits, target, topk=(1, 3, 10))
                print(f'Epoch: {epoch + 1} \t Batch: {batch_index} \t Hit@1: {acc1} \t Hit@3:{acc3} \t Hit@10:{acc10}')

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(self.dataloader)}')
            self.evaluate_save_model(epoch)

    @torch.no_grad()
    def evaluate_epoch(self, epoch):
        total_loss = 0
        acc1s = 0
        acc3s = 0
        acc10s = 0
        for _, (edge_index, triples) in enumerate(self.valid_dataloader):
            if self.device == 'cuda':
                move_to_cuda(edge_index)
                move_to_cuda(triples)
            head, relation, tail = zip(*triples)
            head = torch.tensor(head, dtype=torch.long)
            relation = torch.tensor(relation, dtype=torch.long)
            tail = torch.tensor(tail, dtype=torch.long)
            triples_number = head.shape[0]

            self.optimizer.zero_grad()

            x = self.entity_embeddings.weight
            output = self.model(x, self.adj_matrix)
            
            head_emb = output[head]
            relation_emb = self.relation_embeddings.weight[relation]
            tail_emb = output[tail]

            hr = torch.cat([head_emb, relation_emb], dim=1)

            logits = hr @ tail_emb
            target = torch.arange(triples_number).to(self.device)

            loss = self.criterion(logits, target)
            acc1, acc3 , acc10 = compute_accuracy(logits, target, topk=(1, 3, 10))

            acc1s += acc1.item()
            acc3s += acc3.item()
            acc10s += acc10.item()
            total_loss += loss.item()
        dataloader_len = len(self.valid_dataloader)
        metric_dict = {
            "Hit@1": acc1s/dataloader_len,
            "Hit@3": acc3s/dataloader_len,
            "Hit@10": acc10s/dataloader_len,
            "loss": total_loss/dataloader_len
        }
        print(f'Epoch: {epoch+1} \t valid metric:{metric_dict}')
        return metric_dict
    
    def evaluate_save_model(self, epoch):
        metrics = self.evaluate_epoch(epoch)
        is_best = self.valid_loader and (self.best_metrics is None or metrics['Hit@1'] > self.best_metrics['Hit@1'])
        if is_best:
            self.best_metrics = metrics
        filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)
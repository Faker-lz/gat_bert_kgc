'''
Author: WLZ
Date: 2024-06-03 21:11:37
Description: 
'''
import os 
import torch
import torch.nn as nn
from torch import optim
from metric import compute_accuracy
from logger_config import logger
from torch.utils.data import DataLoader
from knowledge_graph_gat import KnowledgeGraphGAT
from dataset import KnowledgeGraphDataset, load_data
from utils import move_to_cuda, save_checkpoint, delete_old_ckt


class KnowledgeGraphTrainer:
    def __init__(self, all_file_path, train_file_path, valid_file_path, model_dir,layers, entity_dim, hid_dim,relation_dim, dropout, 
                 temperature, alpha, nheads, batch_size=1, lr=0.01, num_epochs=10, device='cuda'):
        self.all_file_path = all_file_path
        self.train_file_path = train_file_path
        self.valid_file_path = valid_file_path
        self.model_dir = model_dir
        self.layers = layers
        self.entity_dim = entity_dim
        self.hid_dim = hid_dim
        self.relation_dim = relation_dim
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs    
        self.device = device
        self.best_metrics = None

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        _, self.all_entity2id, self.all_relation2id = load_data(all_file_path, True)
        self.train_dataset = KnowledgeGraphDataset(train_file_path, self.all_entity2id, self.all_relation2id)
        self.valid_dataset = KnowledgeGraphDataset(valid_file_path, self.all_entity2id, self.all_relation2id)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True)
                
        self.n_entities = len(self.all_entity2id)
        self.n_relations = len(self.all_relation2id)
        
        self.model = KnowledgeGraphGAT(layers, self.n_entities,self.n_relations, entity_dim, relation_dim, 
                                       hid_dim, entity_dim, dropout, alpha, temperature, nheads).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss().to(device)

        
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
                logits, target = self.model(head, relation, tail, adj_matrix)

                loss = self.criterion(logits, target)
                acc1, acc3 , acc10 = compute_accuracy(logits, target, topk=(1, 3, 10))
                logger.info(f'Epoch: {epoch + 1} | Batch: {batch_index} | Loss:{round(loss.item(), 3)} | Hit@1: {round(acc1.item(), 3)} | Hit@3: {round(acc3.item(), 3)} | Hit@10: {round(acc10.item(), 3)}')

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                self.optimizer.zero_grad()
                
            logger.info(f'Epoch {epoch+1}, Loss: {total_loss/len(self.train_dataloader)}')
            self.evaluate_save_model(epoch)

    @torch.no_grad()
    def evaluate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        acc1s = 0
        acc3s = 0
        acc10s = 0
        for _, (adj_matrix, triples) in enumerate(self.valid_dataloader):
            if self.device == 'cuda':
                move_to_cuda(adj_matrix)
                move_to_cuda(triples)
            
            adj_matrix = adj_matrix.squeeze().to(self.device)
            head, relation, tail = zip(*triples)
            logits, target = self.model(head, relation, tail, adj_matrix)
            
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
        logger.info(f'Epoch: {epoch+1} \t valid metric:{metric_dict}')
        return metric_dict
    
    def evaluate_save_model(self, epoch):
        metrics = self.evaluate_epoch(epoch)
        is_best = self.valid_dataloader and (self.best_metrics is None or metrics['Hit@1'] > self.best_metrics['Hit@1'])
        if is_best:
            self.best_metrics = metrics
        filename = '{}/checkpoint_epoch{}.mdl'.format(self.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            # 'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.model_dir),
                       keep=1)
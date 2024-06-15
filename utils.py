'''
Author: WLZ
Date: 2024-06-04 20:48:42
Description: 
'''
import os
import glob
import torch
import shutil

import numpy as np
import torch.nn as nn
from logger_config import logger


def save_checkpoint(state: dict, is_best: bool, filename: str):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.dirname(filename) + '/model_best.mdl')
    shutil.copyfile(filename, os.path.dirname(filename) + '/model_last.mdl')


def delete_old_ckt(path_pattern: str, keep=5):
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        logger.info('Delete old checkpoint {}'.format(f))
        os.system('rm -f {}'.format(f))


def report_num_trainable_parameters(model: torch.nn.Module) -> int:
    assert isinstance(model, torch.nn.Module), 'Argument must be nn.Module'

    num_parameters = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_parameters += np.prod(list(p.size()))
            logger.info('{}: {}'.format(name, np.prod(list(p.size()))))

    logger.info('Number of parameters: {}M'.format(num_parameters // 10**6))
    return num_parameters


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def build_adjacency_matrix(triples, entity2id):
    """
    构建邻接矩阵
    :param triples: 知识图谱三元组列表，每个三元组是(head, relation, tail)
    :param entity2id: 实体到ID的映射字典
    :return: 邻接矩阵
    """
    n_entities = len(entity2id)
    adj_matrix = torch.eye(n_entities, dtype=torch.float32)
    for head, _, tail in triples:
        adj_matrix[tail, head] = 1.0
    return adj_matrix


def build_edge_index(triples, entity2id, is_id=False):
    """
    构建PyTorch Geometric格式的边索引
    :param triples: 知识图谱三元组列表，每个三元组是(head, relation, tail)
    :param entity2id: 实体到ID的映射字典
    :return: PyTorch Geometric格式的边索引
    """
    edges = []
    for head, _, tail in triples:
        if not is_id:
            head_id = entity2id[head]
            tail_id = entity2id[tail]
        else:
            head_id = head
            tail_id = tail
        edges.append([head_id, tail_id])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index



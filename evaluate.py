import os
import torch
import tqdm
from typing import List
from predict import Predictor
from dataset import load_data
from logger_config import logger
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from dataset import KnowledgeGtaphTestDataset
from knowledge_graph_gat import KnowledgeGraphGAT


@torch.no_grad()
def compute_metrics(hr_tensor: torch.tensor,
                    tail_tensor: torch.tensor,
                    target: List[int],
                    entity2id: List[int],
                    topk=10,
                    ):
    assert hr_tensor.size(1) == tail_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity2id)
    assert entity_cnt == tail_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    batch_size = hr_tensor.shape[0]
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        # batch_size * entity_cnt
        batch_score = torch.mm(hr_tensor[start:end, :], tail_tensor.t())
        assert entity_cnt == batch_score.size(1)
        batch_target = target[start:end]

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == batch_score.size(0)
        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            ranks.append(cur_rank)

        topk_scores.extend(batch_sorted_score[:, :topk].tolist())
        topk_indices.extend(batch_sorted_indices[:, :topk].tolist())

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    assert len(topk_scores) == total
    return topk_scores, topk_indices, metrics, ranks

@torch.no_grad()
def predict(all_triple_path, test_triple_path, model_path, device):
    # 完成数据导入
    _, all_entity2id, all_relation2id = load_data(all_triple_path, True)

    predictor = Predictor(device)
    predictor.load(model_path, all_entity2id, all_relation2id)

    hr_vectors, tail_vectors, target = predictor.get_hr_embeddings(all_entity2id, all_relation2id, 1024, 
                                                          test_triple_path)

    topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_vectors, tail_vectors, target, all_entity2id)

    logger.info(f"Metrics: {metrics}")

    # 输出预测结果
    return topk_scores, topk_indices, metrics, ranks
    
if __name__ == '__main__':
    print(predict(r'./data/WN18RR/all.txt', r'./data/WN18RR/test.txt', r'./checkpoint/model_best.mdl', 'cuda'))
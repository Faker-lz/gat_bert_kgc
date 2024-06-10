'''
Author: WLZ
Date: 2024-06-04 21:19:10
Description: 
'''
from logger_config import logger
from train import KnowledgeGraphTrainer


def main():
    trainer = KnowledgeGraphTrainer(
        r'/root/project/wlz/dl/gat_bert_kgc/data/WN18RR/all.txt',
        r'/root/project/wlz/dl/gat_bert_kgc/data/WN18RR/train_graph_parts_20',
        r'/root/project/wlz/dl/gat_bert_kgc/data/WN18RR/valid_graph_parts_3',
        r'/root/project/wlz/dl/gat_bert_kgc/checkpoint',
        3,
        768,
        128,
        32,
        0.2,
        0.05,
        0.2,
        3,
        1,
        0.005,
        50
    )
    logger.info('start trainning ...')
    trainer.train_epoch()

if __name__ == '__main__':
    main()
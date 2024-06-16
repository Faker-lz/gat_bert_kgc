'''
Author: WLZ
Date: 2024-06-04 21:19:10
Description: 
'''
from logger_config import logger
from train import KnowledgeGraphTrainer


def main():
    trainer = KnowledgeGraphTrainer(
        r'./data/WN18RR/all.txt',
        r'./data/WN18RR/train_graph_parts_30',
        r'./data/WN18RR/valid_graph_parts_3',
        r'./checkpoint/dim_768_128_768_epochs_30_split30_layers_2/',
        2,
        768,
        128,
        768,
        32,
        0.2,
        0.05,
        0.2,
        3,
        1,
        0.005,
        30
    )
    logger.info('start trainning ...')
    trainer.train_epoch()

if __name__ == '__main__':
    main()

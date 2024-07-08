'''
Author: WLZ
Date: 2024-06-04 21:19:10
Description: 
'''
from logger_config import logger
from train import KnowledgeGraphTrainer
import wandb_config


def main():
    trainer = KnowledgeGraphTrainer(
        r'./data/WN18RR/all.txt',
        r'./data/WN18RR/train_graph_parts_20',
        r'./data/WN18RR/valid_graph_parts_3',
        r'./checkpoint/beta_class16_dim_768_512_256_epochs_50_split20_layers_3/',
        r'./data/WN18RR/train.txt',
        r'./data/WN18RR/valid.txt',
        3,
        768,
        512,
        256,
        16,
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

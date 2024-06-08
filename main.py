'''
Author: WLZ
Date: 2024-06-04 21:19:10
Description: 
'''
from train import KnowledgeGraphTrainer


def main():
    trainer = KnowledgeGraphTrainer(
        r'E:\study\now\研究生\学习笔记\ML\Graph attention network\data\WN18RR\all.txt',
        r'E:\study\now\研究生\学习笔记\ML\Graph attention network\data\WN18RR\train_graph_parts_50',
        r'E:\study\now\研究生\学习笔记\ML\Graph attention network\data\WN18RR\valid_graph_parts_3',
        4,
        768,
        16,
        0.2,
        0.2,
        3,
        1,
        0.005,
        50
    )
    print('start trainning ...')
    trainer.train_epoch()

if __name__ == '__main__':
    main()
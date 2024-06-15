'''
Author: WLZ
Date: 2024-06-06 22:01:20
Description: 
'''
import os
import random
import numpy as np
from dataset import load_data

def split_knowledge_graph(triples, num_parts):
    random.shuffle(triples)
    parts = np.array_split(triples, num_parts)
    return parts

def save_graph_parts(graph_parts, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, part in enumerate(graph_parts):
        part_file = os.path.join(output_dir, f'part_{i}.txt')
        with open(part_file, 'w') as f:
            for triple in part:
                f.write('\t'.join(triple) + '\n')

def split_and_save_knowledge_graph(input_file, output_dir, num_parts):
    input_file = os.path.join(os.path.dirname(__file__), input_file)
    output_dir = os.path.join(os.path.dirname(__file__), output_dir + f'_{num_parts}')

    triples, _, _ = load_data(input_file, True)
    graph_parts = split_knowledge_graph(triples, num_parts)
    save_graph_parts(graph_parts, output_dir)

if __name__ == '__main__':
    split_and_save_knowledge_graph('data/WN18RR/train.txt', 'data/WN18RR/train_graph_parts', num_parts=30)
    # split_and_save_knowledge_graph(r'data\WN18RR\valid.txt', r'data\WN18RR\valid_graph_parts', num_parts=10)


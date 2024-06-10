'''
Author: WLZ
Date: 2024-06-04 21:20:55
Description: 
'''
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--all-file-path', type=str, required=True, help="Path to the file containing all data")
    parser.add_argument('--train-file-path', type=str, required=True, help="Path to the training data file")
    parser.add_argument('--valid-file-path', type=str, required=True, help="Path to the validation data file")
    parser.add_argument('--entity_dim', type=int, required=True, help="Dimension of the entity embeddings")
    parser.add_argument('--relation_dim', type=int, required=True, help="Dimension of the relation embeddings")
    parser.add_argument('--hid_dim')
    parser.add_argument('--dropout', type=float, required=True, help="Dropout rate")
    parser.add_argument('--alpha', type=float, required=True, help="Alpha value for LeakyReLU")
    parser.add_argument('--nheads', type=int, required=True, help="Number of attention heads")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for training, 'cuda' or 'cpu'")
    parser.add_argument('--t', type=float, default=0.05, help='temperature of infoENCE')



    args = parser.parse_args()
    return args

args = get_args()

if __name__ == "__main__":
    args = get_args()
    print(args)

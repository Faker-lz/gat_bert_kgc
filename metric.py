'''
Author: WLZ
Date: 2024-06-04 20:21:32
Description: 
'''
import torch
from typing import List

def compute_accuracy(ouput:torch.tensor, target:torch.tensor, topk=(1,))-> List[torch.tensor]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = ouput.shape[0]

        _, pred = torch.topk(ouput, maxk, 1, True, True)
        pred = pred.t()
        correct = torch.eq(pred, target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    # 假设模型输出和真实标签如下
    output = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.2, 0.5]])
    target = torch.tensor([2, 0])

    # 计算 top-1 和 top-2 准确率
    print(compute_accuracy(output, target, topk=(1, 2)))

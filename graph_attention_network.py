'''
Author: WLZ
Date: 2024-06-02 15:07:29
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F 

class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha, concat=True) -> None:
        super().__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # self.W = torch.nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)
        # self.a = torch.nn.Linear(2*input_dim, 1, bias=False)
        # 使用均匀分布初始化系数矩阵
        self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2*self.out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.LeakReLU = nn.LeakyReLU(self.alpha)

    def forward(self, features: torch.tensor, adj: torch.tensor):
        # N = features.size(0)
        # N * O
        h = torch.mm(features, self.W)

        # Matrix compute but ineffective
        # N * F -> (N * N) * F
        # h_1 = h.repeat(1, N).view(N * N, -1)
        # h_2 = h.repeat(N, 1)

        # # N * N * (2 * O)
        # h_cat = torch.cat([h_1, h_2], dim=1).view(N, N, 2*self.out_features)
        
        # N * N * (2 * O) @ (2 * O) * 1 -> N * N
        # e = self.LeakReLU(torch.matmul(h_cat, self.a).squeeze())
    
        e = self.effective_compute_e(h)

        attention = torch.full_like(e, -9e15)
        attention[adj > 0] = e[adj > 0]
        attention = torch.softmax(attention, dim=-1)


        # inf_matrix = -9e15 * torch.ones_like(e)
        # attention = torch.softmax(torch.where(adj > 0, e, inf_matrix), dim=-1)

        # attention = F.dropout(attention, self.dropout, training=self.training)

        h = torch.mm(attention, h)
        
        # 如果不是最后一层则使用激活函数激活
        if self.concat:
            h = F.elu(h)
        return h
    
    def effective_compute_e(self, h):
        h1 = torch.matmul(h, self.a[:self.out_features, :])
        h2 = torch.matmul(h, self.a[self.out_features: , :])
        e = h1 + h2.T
        return self.LeakReLU(e)
    

class MultiHeadGAT(nn.Module):
    def __init__(self, nfeat, nhid, noutfeat, dropout, alpha, nheads) -> None:
        super().__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.noutfeat = noutfeat
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads

        self.attentions = nn.ModuleList(
            [GraphAttentionLayer(input_dim=nfeat, output_dim=nhid,
                                 dropout=dropout, alpha=alpha)
                                 for _ in range(nheads)]
        )
        
        self.out_layer = GraphAttentionLayer(input_dim=nheads * nhid, output_dim=noutfeat,
                                             dropout=dropout, alpha=alpha, concat=False)
    
    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        # 多头注意力特征拼接
        x = torch.cat([attention(x, adj) for attention in self.attentions] ,dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_layer(x, adj)
        # 分类任务需要注释掉
        # x = F.dropout(x, self.dropout, training=self.training)
        # # 节点分类任务损失
        # x = F.log_softmax(F.elu(x), dim=1)
        return x

        
class MultiLayerGAT(nn.Module):
    def __init__(self, nlayer, nfeat, nhid, noutfeat, dropout, alpha, nheads) -> None:
        super().__init__()
        self.nlayer = nlayer
        self.nfeat = nfeat
        self.nhid = nhid
        self.noutfeat = noutfeat
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads

        self.gats = nn.ModuleList()
        self.gats.append(MultiHeadGAT(nfeat=nfeat, nhid=nhid, noutfeat=nhid,
                                    nheads=nheads, alpha=alpha, dropout=dropout))
        
        for _ in range(nlayer - 1):
            self.gats.append(MultiHeadGAT(nfeat=nhid, nhid=nhid, noutfeat=nhid, 
                                          nheads=nheads, alpha=alpha, dropout=dropout))

        self.out_layer = MultiHeadGAT(nfeat=nhid, nhid=nhid, noutfeat=noutfeat, nheads=nheads, 
                                      alpha=alpha, dropout=dropout)
        
    def forward(self, x, adj):
        for gat in self.gats:
            x = gat(x, adj)
        x = self.out_layer(x, adj)
        return x


if __name__ == '__main__':
    x = torch.randn(6, 768).to('cuda')
    adj = torch.ByteTensor([
        [0,1,1,0,0,0],
        [1,0,1,0,0,0],
        [1,1,0,1,0,0],
        [0,0,1,0,1,1],
        [0,0,0,1,0,1],
        [0,0,0,1,1,0]
    ]).to('cuda')
    # my_gat_layer = MultiHeadGAT(10, 5, 3, 0.2, 0.2,10)
    my_gat_layer = MultiLayerGAT(10, 768, 128, 768, 0.2, 0.2, 10).to('cuda')
    result = my_gat_layer(x, adj)
    print(result)
    print(F.normalize(result, dim=1))
        
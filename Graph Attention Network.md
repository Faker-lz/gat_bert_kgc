# Graph Attention Network

## 计算原理
其本质是节点与其相连的邻居节点计算Attention之后，用Attention矩阵加权计算该节点邻居节点的特征聚合。具体原理如图所示：
![alt text](image.png)
其中左图表示某个节点的attention计算机制$a(W\vec{h_i}, W\vec{h_j})$，右图表示多头的注意力机制(图中示例有三个不同的注意力头)，不同颜色的箭头代表着独立的不同类型注意力头。

### 计算流程
计算流程以GAT中的单层为例剖析，其中每个网络层的输入$\vec{h}=\{\vec{h_1}, \vec{h_2},..., \vec{h_N}\}$，输出为经过邻居节点注意力机制聚合后的特征$\vec{h'=\{\vec{h_1'}, \vec{h_2'}, ..., \vec{h_N'}\}}$，其中N代表节点数量，$\vec{h} \in \mathcal{R}^F$,$\vec{h'} \in \mathcal{R}^{F'}$

1. 计算节点间的注意力矩阵
    - 计算attention系数: $e_{ij}=a(W\vec{h_i}, W\vec{h_j})$，表示第$j$个节点的特征对第$i$个节点的重要性程度。
    - 计算attention矩阵:$a_{ij}=softmax(e_{ij})=\frac{exp(e_{ij})}{\sum_{k \in \mathcal{N}_i}exp(e_{ik})}$，使用softmax函数将attention系数规范化。
    > 注意，在实验中，attention系数的计算还增加了LeakReLU函数，所以具体计算公式更正如下：
    > $$\alpha_{ij}=\frac{\exp\left(\mathrm{LeakyReLU}\left(\vec{\mathbf{a}}^T[\mathbf{W}\vec{h}_i\|\mathbf{W}\vec{h}_j]\right)\right)}{\sum_{k\in\mathcal{N}_i}\exp\left(\mathrm{LeakyReLU}\left(\vec{\mathbf{a}}^T[\mathbf{W}\vec{h}_i\|\mathbf{W}\vec{h}_k]\right)\right)}$$
2. 使用attention矩阵融合节点特征。
    - 单头注意力融合： $\vec{h}_i'=\sigma\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\mathbf{W}\vec{h}_j\right)$。
    - 为了使得self-attention学习过程更加稳定，引入了多头注意力机制:$\vec{h}_i'=\prod\limits_{k=1}^K\sigma\left(\sum\limits_{j\in\mathcal{N}_i}\alpha_{ij}^k\mathbf{W}^k\vec{h}_j\right)$，其中$\alpha_{ij}^k$是第K个注意力头的注意力矩阵，$W^k$是相关的线性特征提取矩阵。
    - 此外，如果对于最后一个多头注意力层，简单拼接不利于后续特征提取以及下游任务计算，采用mean方法融合多头注意力：
    $$\vec{h}_i'=\sigma\left(\frac{1}{K}\sum_{k=1}^K\sum_{j\in\mathcal{N}_i}\alpha_{ij}^k\mathbf{W}^k\vec{h}_j\right)$$

## 代码实现
见模型文件`graphAttentionNetwork.py`
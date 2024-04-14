import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

class MyScore(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # ① From the perspective of graph structure
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        # ② From the perspective of node features
        self.mlp = nn.Linear(in_channels, 1, bias=False)
        # ③ From the joint perspective of structure and features
        self.linear = nn.Linear(in_channels, 1, bias=False)
        self.gnn = GCNConv(in_channels, 1, bias=False)  # GCNConv layer for joint perspective
        # Attention mechanism for fusion
        self.attention = nn.Linear(3, 3)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # ① From the perspective of graph structure
        score1 = torch.sigmoid(self.alpha * torch.sqrt(degree(edge_index[1], num_nodes)) + self.beta)
        # ② From the perspective of node features
        score2 = torch.sigmoid(self.mlp(x)).squeeze()
        # ③ From the joint perspective of structure and features
        score3 = torch.sigmoid(self.gnn(x, edge_index) + self.linear(x)).squeeze()
        # Fusion of scores
        scores = torch.stack([score1, score2, score3], dim=1)
        weight = self.attention(scores)
        weight = torch.softmax(weight, dim=1)
        fitness = torch.sum(weight * scores, dim=1)
        return fitness

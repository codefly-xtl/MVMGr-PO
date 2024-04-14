import torch.nn as nn
from torch_geometric.nn import GCN, GAT, GraphSAGE
from node_classification_main.mypool import Pooling

class Coarsen(nn.Module):
    def __init__(self, feature_size, hidden_size, coarse_layers, num_layers, ego_range, activate_function, dropout,
                 head, GNN):
        super().__init__()
        self.activate_function = activate_function
        self.dropout = dropout
        self.coarse_layers = coarse_layers
        self.ego_range = ego_range
        self.num_layers = num_layers
        self.head = head
        self.gnns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.GNN = GNN
        for i in range(coarse_layers):
            input_size = feature_size if i == 0 else hidden_size
            if GNN == "GCN":
                self.gnns.append(GCN(input_size, hidden_size, self.num_layers, hidden_size))
            elif GNN == "GAT":
                self.gnns.append(GAT(input_size, hidden_size, self.num_layers, hidden_size, heads=head))
            elif GNN == "GraphSAGE":
                self.gnns.append(GraphSAGE(input_size, hidden_size, self.num_layers, hidden_size))
            self.pools.append(Pooling(hidden_size, ego_range, activate_function))

    def forward(self, x, edge_index):
        x_list = []
        edge_index_list = [edge_index]
        S_list = []
        for i in range(self.coarse_layers):
            x = self.gnns[i](x, edge_index)
            x = self.activate_function(x)
            x_list.append(x)
            x, edge_index, S, _ = self.pools[i](x, edge_index)
            S_list.append(S)
            edge_index_list.append(edge_index)
        return x, x_list, edge_index_list, S_list

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.nn import GCN, GraphSAGE, GAT


class Refine(nn.Module):
    def __init__(self, hidden_size, output_size, coarse_layers, activate_function, num_layers, dropout, GNN, head):
        super().__init__()
        self.dropout = dropout
        self.activate_function = activate_function
        self.coarse_layers = coarse_layers
        self.num_layers = num_layers
        self.refinement = torch.nn.ModuleList()

        # Initialize refinement layers based on the specified GNN
        for i in range(coarse_layers):
            if GNN == "GCN":
                self.refinement.append(GCN(hidden_size, hidden_size, self.num_layers, hidden_size))
            elif GNN == "GAT":
                self.refinement.append(GAT(hidden_size, hidden_size, self.num_layers, hidden_size, heads=head))
            elif GNN == "GraphSAGE":
                self.refinement.append(GraphSAGE(hidden_size, hidden_size, self.num_layers, hidden_size))

        # Add final refinement layer
        if GNN == "GCN":
            self.refinement.append(GCN(hidden_size, hidden_size, self.num_layers, output_size))
        elif GNN == "GAT":
            self.refinement.append(GAT(hidden_size, hidden_size, self.num_layers, output_size, heads=head))
        elif GNN == "GraphSAGE":
            self.refinement.append(GraphSAGE(hidden_size, hidden_size, self.num_layers, output_size))

    def forward(self, x, x_list, edge_index_list, S_list):
        for i in range(self.coarse_layers):
            idx = self.coarse_layers - i
            # Apply refinement layer
            x = self.refinement[i](x, edge_index_list[idx])
            x = self.activate_function(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Aggregate coarse features
            x = S_list[idx - 1] @ x + x_list[idx - 1]
        # Apply final refinement layer
        x = self.refinement[-1](x, edge_index_list[0])
        return x

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid, Coauthor
from model import MyModel
from utils.utils import random_split

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # Load dataset based on the provided dataset name
        if args.dataset in ['Pubmed', 'Cora', 'Citeseer']:
            self.data = Planetoid(root='../data', name=args.dataset, split=self.args.split)
            self.graph = self.data[0].to(self.device)
        elif args.dataset in ['CS', 'Physics']:
            self.data = Coauthor(root='../data', name=args.dataset)
            self.graph = self.data[0].to(self.device)
            # Split the dataset into train, validation, and test sets
            self.graph.train_mask, self.graph.val_mask, self.graph.test_mask = random_split(self.graph, 20, 30)

        self.feature_size = self.graph.num_node_features
        self.num_classes = self.data.num_classes
        self.loss = nn.CrossEntropyLoss()
        self.link_loss = nn.BCEWithLogitsLoss()
        self.dropout = args.dropout
        self.lr = args.lr
        self.epochs = args.epochs
        self.ego_range = args.ego_range
        self.weight_decay = args.weight_decay
        self.hidden_size = args.hidden_size
        self.GNN = args.GNN
        self.head = args.head
        self.coarse_layers = args.coarse_layers
        self.num_layers = args.num_layers
        self.activate = args.activate
        self.negative_slope = args.negative_slope
        self.model = None
        self.optimizer = None

        # Print dataset information
        print(f"Training set size: {torch.sum(self.graph.train_mask)}, Validation set size: {torch.sum(self.graph.val_mask)}, Test set size: {torch.sum(self.graph.test_mask)}")

    def set_model(self):
        # Initialize the model
        self.model = MyModel(self.feature_size, self.hidden_size, self.num_classes, self.coarse_layers, self.num_layers,
                             self.ego_range, self.activate, self.negative_slope, self.dropout, self.head, self.GNN).to(self.device)

    def set_optimizer(self):
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_one_epoch(self):
        # Train the model for one epoch
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.graph.x, self.graph.edge_index)
        loss = self.loss(out[self.graph.train_mask], self.graph.y[self.graph.train_mask])
        loss.backward()
        self.optimizer.step()

        # Calculate train accuracy
        pred = out[self.graph.train_mask].max(dim=1)[1]
        correct = pred.eq(self.graph.y[self.graph.train_mask]).sum().item()
        num_train_nodes = torch.sum(self.graph.train_mask).item()
        return loss, correct / num_train_nodes

    def test(self, mask):
        # Test the model
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.graph.x, self.graph.edge_index)
            pred = out[mask].max(dim=1)[1]
            correct = pred.eq(self.graph.y[mask]).sum().item()
            num_nodes = torch.sum(mask).item()
            return correct / num_nodes

    def train(self):
        # Train the model
        self.set_model()
        self.set_optimizer()
        best_val_acc = 0

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
            val_acc = self.test(self.graph.val_mask)

            # Save the best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.args.dataset + self.GNN + 'best_model.pth')

            print(f"Epoch: [{epoch}/{self.args.epochs}], Loss: {train_loss}, Train Accuracy: {round(train_acc, 4)}, Validation Accuracy: {round(val_acc, 4)}")

        # Load the best model
        self.set_model()
        self.model.load_state_dict(torch.load(self.args.dataset + self.GNN + 'best_model.pth'))
        test_acc = self.test(self.graph.test_mask)
        print(f"Test Accuracy: {round(test_acc, 4)}")

        return test_acc

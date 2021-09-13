import torch
import torch.nn.functional as F 
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap


"""
Graph-Convolutional Neural Network
"""

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(21)
        self.conv1 = GCNConv(30, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, int(hidden_channels/2))
        self.conv3 = GCNConv(int(hidden_channels/2), int(hidden_channels/4))
        self.linear = Linear(int(hidden_channels/4), 1)

    def forward(self, data, edge_index, batch):
        x, targets = data.x, data.y
        # 1. Obtain the node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        # 2. Aggregating message passing/embeddings
        x = gap(x, batch)
        
        # 3. Apply the final regressor
        x = F.dropout(x, p=0.25, training=self.training)

        # model output from forward and loss 
        out = self.linear(x)  
        loss = torch.nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))

        out = torch.sigmoid(out) # converting out proba in range [0, 1]
        return out, loss


class GAT(torch.nn.Module):
    pass
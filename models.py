import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv

class FirstNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(FirstNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.conv4 = GCNConv(64, num_classes)
        self.dropout = 0.1

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = nn.Dropout(self.dropout)(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = nn.Dropout(self.dropout)(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = nn.Dropout(self.dropout)(x)
        x = self.conv4(x, edge_index)

        x = pyg_nn.global_max_pool(x, batch)

        return F.log_softmax(x, dim=1)


class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):

        self.dropout = float(args.dropout)
        self.num_layers = int(args.num_layers)

        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        self.batchnorm_layers.append(nn.BatchNorm1d(hidden_dim))
        assert (self.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(self.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))
            self.batchnorm_layers.append(nn.BatchNorm1d(hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(3*hidden_dim, 3*hidden_dim), nn.Dropout(self.dropout),
            nn.Linear(3*hidden_dim, output_dim))


    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv
        elif model_type == 'GAT':
            return pyg_nn.GATConv

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch = batch.to(x.device)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.batchnorm_layers[i](x)
            x = F.dropout(x, self.dropout, training=self.training)  # N x embedding size

        # concatenate max_pool, mean_pool and embedding of first node (i.e. the news root)
        x1 = pyg_nn.global_max_pool(x, batch) # shape batch_size * embedding size
        x2 = pyg_nn.global_mean_pool(x, batch)

        batch_size = x1.size(0)
        indices_first_nodes = [(data.batch == i).nonzero()[0] for i in range(batch_size)]
        x3 = x[indices_first_nodes, :]

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)



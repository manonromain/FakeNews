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
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))


        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)  # N x embedding size
        x = pyg_nn.global_max_pool(x, batch)

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        self.agg_lin = nn.Sequential(nn.Linear(in_channels + out_channels, out_channels), nn.ReLU())
                                 #nn.Linear(in_channels + out_channels, out_channels), nn.ReLU()) # MLP
        self.lin = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU()) # TODO

        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, in_channels]
        # edge_index has shape [2, E]

        x_j = self.lin(x_j)  # TODO

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]

        concat = torch.cat([aggr_out, x], dim=-1)
        aggr_out = self.agg_lin(concat)  # TODO

        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=1)

        return aggr_out


class GAT(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, num_heads=1, concat=True,
                 dropout=0, bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat 
        self.dropout = dropout

        self.lin = nn.Linear(self.in_channels, self.heads * out_channels) # TODO
        self.att = nn.Parameter(torch.Tensor(self.heads, 2 * out_channels)) # TODO

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, size=None):
        
        x = self.lin(x) # TODO
        # Start propagating messages.
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        #  Constructs messages to node i for each edge (j, i).

        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        concat = torch.cat((x_i, x_j), dim=2)
        concat = concat.view(-1, self.heads * 2 * self.out_channels)
        att_result = F.leaky_relu(torch.mm(concat, self.att.t()), negative_slope=0.02)
        alpha = F.softmax(att_result, dim=1) # TODO

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        # Updates node embeddings.
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv

from dataset import data_twitter

BATCH_SIZE = 32


def train(dataset="twitter15"):
    dataset = data_twitter(dataset)
    data_loader = torch_geometric.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    model = Net(4794, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        epoch_loss = 0
        for batch in data_loader:
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            epoch_loss += loss.sum().item()
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "loss:", epoch_loss / len(data_loader))

    model.eval()
    correct = 0
    n_samples = 0
    for batch in data_loader:
        _, pred = model(batch).max(dim=1)
        correct += float(pred.eq(batch.y).sum().item())
        n_samples += len(batch.y)
    acc = correct / n_samples
    print('Accuracy: {:.4f}'.format(acc))
    print('Correct: {}'.format(correct))
    print('Sur: {}'.format(n_samples))
    return


class Net(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(Net, self).__init__()
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


if __name__ == "__main__":
    train("twitter15")

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv

from models import FirstNet
from dataset import DatasetBuilder


def train(dataset, args):

    # Loading dataset
    dataset = DatasetBuilder(dataset).create_dataset()
    data_loader = torch_geometric.data.DataLoader(dataset, batch_size=args.batch_size)
    
    # Setting up model
    model = FirstNet(4794, 4)
    
    # Training phase
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    model.train()
    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        epoch_loss = 0
        for batch in data_loader:
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            epoch_loss += loss.sum().item()
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "loss:", epoch_loss / len(data_loader))

    # Evaluation on the TRAINING SET 
    model.eval()
    correct = 0
    n_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            _, pred = model(batch).max(dim=1)
            correct += float(pred.eq(batch.y).sum().item())
            n_samples += len(batch.y)
    acc = correct / n_samples
    print('Accuracy: {:.4f}'.format(acc))
    print('True_positives {} over {}'.format(correct, n_samples))
    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the graph network.')
    parser.add_argument('dataset', choices=["twitter15", "twitter16"],  
                    help='Training dataset', default="twitter15")
    parser.add_argument('--lr', default=0.01,
                    help='learning rate')
    parser.add_argument('--num_epochs', default=200,
                    help='Number of epochs')
    parser.add_argument('--batch_size', default=32,
                    help='Batch_size')

    args = parser.parse_args()
    train(args.dataset, args)

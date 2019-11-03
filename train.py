import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter

from models import FirstNet, GNNStack
from dataset import DatasetBuilder


def train(dataset, args):

    # Loading dataset
    dataset_builder = DatasetBuilder(dataset, only_binary=True)
    datasets = dataset_builder.create_dataset()
    train_data_loader = torch_geometric.data.DataLoader(datasets["train"], batch_size=args.batch_size)
    val_data_loader = torch_geometric.data.DataLoader(datasets["val"], batch_size=args.batch_size)
    test_data_loader = torch_geometric.data.DataLoader(datasets["test"], batch_size=args.batch_size)

    # Setting up model
    model = FirstNet(dataset_builder.number_of_features, 2)

    # Tensorboard logging
    log_dir = os.path.join("logs", args.exp_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    train_writer = SummaryWriter(os.path.join(log_dir, "train"))
    val_writer = SummaryWriter(os.path.join(log_dir, "val"))
    test_writer = SummaryWriter(os.path.join(log_dir, "test"))

    # Checkpoints
    checkpoint_dir = os.path.join("checkpoints", args.exp_name)
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.isfile(checkpoint_path):
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        epoch_ckp = 0
        global_step = 0
    else:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch_ckp = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        print("Restoring previous model at epoch", epoch_ckp)

    # Training phase
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(epoch_ckp, epoch_ckp + args.num_epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0
        for batch in train_data_loader:
            #import pdb; pdb.set_trace()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            epoch_loss += loss.sum().item()

            # Optimization
            loss.backward()
            optimizer.step()

            # TFBoard logging
            train_writer.add_scalar("loss", loss.mean(), global_step)
            global_step += 1

        # Saving model at the end of each epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "epoch_loss": epoch_loss / len(train_data_loader),
            "global_step": global_step
        }
        torch.save(checkpoint, checkpoint_path)
        print("epoch", epoch, "loss:", epoch_loss / len(train_data_loader))

        # Evaluation on the validation set 
        model.eval()
        correct = 0
        n_samples = 0
        with torch.no_grad():
            for batch in train_data_loader:
                _, pred = model(batch).max(dim=1)
                correct += float(pred.eq(batch.y).sum().item())
                n_samples += len(batch.y)
        acc = correct / n_samples
        train_writer.add_scalar("Accuracy", acc, global_step)
        print('Training accuracy: {:.4f}'.format(acc))

        # Evaluation on the validation set 
        model.eval()
        correct = 0
        n_samples = 0
        with torch.no_grad():
            for batch in val_data_loader:
                _, pred = model(batch).max(dim=1)
                correct += float(pred.eq(batch.y).sum().item())
                n_samples += len(batch.y)
        acc = correct / n_samples
        val_writer.add_scalar("Accuracy", acc, global_step)
        print('Validation accuracy: {:.4f}'.format(acc))
        #print('True_positives {} over {}'.format(correct, n_samples))
    return


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    parser = argparse.ArgumentParser(description='Train the graph network.')
    parser.add_argument('dataset', choices=["twitter15", "twitter16"],
                    help='Training dataset', default="twitter15")
    parser.add_argument('--lr', default=0.01,
                    help='learning rate')
    parser.add_argument('--num_epochs', default=200,
                    help='Number of epochs')
    parser.add_argument('--num_layers', default=2, type=int,
                    help='Number of layers')
    parser.add_argument('--dropout', default=0.2,
                    help='Model type for GNNStack')
    parser.add_argument('--model_type', default="GAT",
                    help='Model type for GNNStack')
    parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch_size')
    parser.add_argument('--exp_name', default="default",
                    help="Name of experiment - different names will log in different tfboards and restore different models")
    args = parser.parse_args()
    train(args.dataset, args)

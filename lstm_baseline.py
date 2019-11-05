import argparse
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import random
from dataset import DatasetBuilder
from utils import preprocess_sequences_to_fixed_len, standardize_and_turn_tensor
from itertools import chain
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=["twitter15", "twitter16"],
                    help='Training dataset', default="twitter15")
parser.add_argument('--lr', default=0.01,
                    help='learning rate')
parser.add_argument('--num_epochs', default=250,
                    help='Number of epochs')
parser.add_argument('--num_lstm_layers', default=2, type=int,
                    help='Number of lstm layers')
parser.add_argument('--num_linear_layers', default=1, type=int,
                    help='Number of mlp layers')
parser.add_argument('--hidden_size', default=12, type=int,
                    help='Hidden size')
parser.add_argument('--dropout', default=0.8)
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch_size')
parser.add_argument('--debug', default=1, type=int,
                    help='In debugging, we train on val')
parser.add_argument('--test_on_train', default=1, type=int,
                    help='overfit on train?')
parser.add_argument('--verbose', default=0, type=int,
                    help='If verbose, print running loss at every step')
parser.add_argument('--cap_len', default=40, type=int,
                    help='Cap on the lenght of the sequences passed to the LSTM')
parser.add_argument('--exp_name', default="LSTM_default",
                    help="Name of experiment - different names will log in different tfboards and restore different models")


class SeqDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, Y):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.Y.size(0)

    def __getitem__(self, idx):
        sample = {'sequence': self.X[idx, :, :], 'label': self.Y[idx]}
        return sample

def seq_data_to_dataset(seq_data, cap_len, num_features):
    X, idx_removed = preprocess_sequences_to_fixed_len(seq_data, cap_len, num_features)
    X = standardize_and_turn_tensor(X)
    Y = torch.from_numpy(np.concatenate([x_y[1] for ix, x_y in enumerate(seq_data) if ix not in idx_removed]))
    print(f"generated tensor datasets of size: X{X.size()}, Y{Y.size()}")
    return SeqDataset(X, Y)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, seq_size, h_size, n_classes=2, n_lstm_layers=1, n_linear_layers_hidden=0, dropout=0.):
        super(LSTMClassifier, self).__init__()

        # self.bn = nn.BatchNorm1d(seq_size, affine=False)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=h_size, num_layers=n_lstm_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=False)
        self.linear = nn.Sequential(
            *chain(*[(nn.Linear(h_size, h_size), nn.ReLU()) for _ in range(n_linear_layers_hidden)]),
            nn.Linear(h_size, n_classes))
        self.n_classes = n_classes

    def forward(self, seq):
        """
        input = seq = torch FloatTensor of size (B, Seqlen, Input_size)
        output is of size (B, hidden_size * num layers) -> concatenation of the last hidden state over the LSTM layers
        returns = the logits of this output passed to a linear layer
        """
        # seq = self.bn(seq)
        out, (_, _) = self.lstm(seq)
        return self.linear(out.mean(1))


def dataset_iterator(dataset, batch_size=32, shuffle=True):
    """
    :param dataset: sequential dataset as returned by create_dataset
    :param batch_size
    :param cap_len: if not None, the cap on the sequence lengths you want to put
    :param shuffle: if you want to shuffle the dataset at the beginning of each epoch
    :return: yield batches for the training, each of the form [(sequence:TorchTensor, sequence_label:TorchTensor)] with len <=batch_size
    each sequence Tensor is of size (<=cap_len or length of the tweets sequence, num_features)
    """
    if shuffle:
        random.shuffle(dataset)
    count_sampled = 0
    n = len(dataset)
    while (count_sampled < n):
        yield dataset[count_sampled:count_sampled + batch_size]
        count_sampled += batch_size


def train(args, model, optim, train_loader, test_loader=None, baseline_accuracy=0.5):
    # Tensorboard logging
    log_dir = os.path.join("logs", args.exp_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    train_writer = SummaryWriter(os.path.join(log_dir, "train"))

    # Checkpoints
    checkpoint_dir = os.path.join("checkpoints", args.exp_name)
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.isfile(checkpoint_path):
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    epoch_ckp = 0
    global_step = 0

    # Training phase
    loss_function = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(epoch_ckp, epoch_ckp + args.num_epochs):

        # lr = args.lr * (0.99) ** epoch
        # for param_group in optim.param_groups:
        #     param_group['lr'] = lr

        model.train()
        epoch_loss = 0
        running_loss = 0
        for ix, batch in enumerate(train_loader):#enumerate(dataset_iterator(train_loader, batch_size=args.batch_size, shuffle=True)):
            # import pdb;
            # pdb.set_trace()
            sequences, ys = batch['sequence'], batch['label']
            # ys = torch.cat(ys)
            # print(torch.cat(sequences).size())
            # sequences = pad_sequence(sequences, batch_first=True)
            # print(sequences[0, 0, :])
            # import pdb;
            # pdb.set_trace()
            logits = model(sequences.float())#.float())
            loss = loss_function(logits, ys)

            # Optimization
            optim.zero_grad()
            loss.backward()
            optim.step()

            # TFBoard logging
            train_writer.add_scalar("loss", loss.item(), global_step)
            global_step += 1

            # Printing running loss
            epoch_loss += loss.item()
            if not ix:
                running_loss = loss.item()
            else:
                running_loss = running_loss * 0.5 + loss.item() * 0.5
            if args.verbose:
                print(f"Step {ix + 1}, running loss: {running_loss:.4f}")

        # Saving model at the end of each epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "epoch_loss": args.batch_size * epoch_loss / len(train_loader),
            "global_step": global_step
        }
        torch.save(checkpoint, checkpoint_path)
        print("epoch", epoch, "loss:", args.batch_size * epoch_loss / len(train_loader))

        # Evaluation on the TRAINING SET
        if (epoch + 1) % 5 == 0:
            test_loader = train_loader if test_loader is None else test_loader
            model.eval()
            correct = 0
            n_samples = 0
            with torch.no_grad():
                for batch in test_loader:#dataset_iterator(test_loader, batch_size=args.batch_size, shuffle=True):
                    sequences, ys = batch['sequence'], batch['label']
                    # ys = torch.cat(ys)
                    # sequences = pad_sequence(sequences, batch_first=True)
                    _, pred = model(sequences.float()).max(dim=1)
                    # print(pred)
                    # import pdb;
                    # pdb.set_trace()
                    correct += float(pred.eq(ys).sum().item())
                    n_samples += pred.size(0)
            acc = correct / n_samples
            train_writer.add_scalar("Accuracy", acc, global_step)
            print('Accuracy: {:.4f}, vs random accuracy on train: {:.4f}'.format(acc, baseline_accuracy))
    return


if __name__ == "__main__":
    args = parser.parse_args()
    # Loading dataset

    dataset_builder = DatasetBuilder(args.dataset, only_binary=True, time_cutoff=1500)
    full_dataset = dataset_builder.create_dataset(dataset_type="sequential", standardize_features=False)
    val_dataset = full_dataset['val']

    if args.debug:
        train_dataset = val_dataset
    else:
        train_dataset = full_dataset['train']

    train_dataset = seq_data_to_dataset(train_dataset, cap_len=args.cap_len, num_features=11)
    val_dataset = seq_data_to_dataset(val_dataset, cap_len=args.cap_len, num_features=11)
    train_loader = DataLoader(dataset = train_dataset, batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(dataset = val_dataset, batch_size=args.batch_size,shuffle=True) if not args.test_on_train else train_loader#to change if different number of features

    baseline_acc = train_dataset.Y.float().mean().item()
    baseline_acc = max(1 - baseline_acc, baseline_acc)
    print(f"Baseline accuracy on train is: {baseline_acc:.2f}")
    # Setting up model
    model = LSTMClassifier(input_size=11,
                           seq_size=args.cap_len,
                           h_size=args.hidden_size,
                           n_lstm_layers=args.num_lstm_layers,
                           n_linear_layers_hidden=args.num_linear_layers-1,
                           dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(args, model, optimizer, train_loader, val_loader, baseline_acc)

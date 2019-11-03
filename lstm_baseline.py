import argparse
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import random
from dataset import DatasetBuilder

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=["twitter15", "twitter16"],
                    help='Training dataset', default="twitter15")
parser.add_argument('--lr', default=0.01,
                    help='learning rate')
parser.add_argument('--num_epochs', default=200,
                    help='Number of epochs')
parser.add_argument('--num_layers', default=2, type=int,
                    help='Number of layers')
parser.add_argument('--hidden_size', default=128, type=int,
                    help='Hidden size')
parser.add_argument('--dropout', default=0.)
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch_size')
parser.add_argument('--debug', default=1, type=int,
                    help='In debugging, we train on val and try to overfit it')
parser.add_argument('--verbose', default=0, type=int,
                    help='If verbose, print running loss at every step')
parser.add_argument('--cap_len', default=50, type=int,
                    help='Cap on the lenght of the sequences passed to the LSTM')
parser.add_argument('--exp_name', default="LSTM_default",
                    help="Name of experiment - different names will log in different tfboards and restore different models")


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, h_size, n_classes=2, n_layers=2, dropout=0.):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=h_size, num_layers=n_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=False)
        self.linear = nn.Linear(h_size * n_layers, n_classes)
        self.n_classes = n_classes
        self.n_layers = n_layers

    def forward(self, seq):
        """
        input = seq = torch FloatTensor of size (B, Seqlen, Input_size)
        output is of size (B, hidden_size * num layers) -> concatenation of the last hidden state over the LSTM layers
        returns = the logits of this output passed to a linear layer
        """
        _, (h_n, _) = self.lstm(seq)
        output = torch.cat([h_n[i, :, :] for i in range(self.n_layers)], dim=-1)
        return self.linear(output)


def dataset_iterator(dataset, batch_size=32, shuffle=True, cap_len=None):
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
        batch_to_yield = dataset[count_sampled:count_sampled + batch_size]
        if cap_len is None:
            yield batch_to_yield
        else:
            yield [[elt[0][:cap_len, :], elt[1]] for elt in batch_to_yield]
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
        model.train()
        epoch_loss = 0
        running_loss = 0
        for ix, batch in enumerate(dataset_iterator(train_loader, batch_size=args.batch_size, shuffle=True,
                                                    cap_len=None if not args.cap_len else args.cap_len)):
            # import pdb;
            # pdb.set_trace()
            sequences, ys = list(zip(*batch))
            ys = torch.cat(ys)
            sequences = pad_sequence(sequences, batch_first=True)
            logits = model(sequences.float())
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
        test_loader = train_loader if test_loader is None else test_loader
        model.eval()
        correct = 0
        n_samples = 0
        with torch.no_grad():
            for batch in dataset_iterator(train_loader, batch_size=args.batch_size, shuffle=True,
                                                    cap_len=None if not args.cap_len else args.cap_len):
                sequences, ys = list(zip(*batch))
                ys = torch.cat(ys)
                sequences = pad_sequence(sequences, batch_first=True)
                _, pred = model(sequences.float()).max(dim=1)
                # import pdb;
                # pdb.set_trace()
                correct += float(pred.eq(ys).sum().item())
                n_samples += pred.size(0)
        acc = correct / n_samples
        train_writer.add_scalar("Accuracy", acc, global_step)
        print('Accuracy: {:.4f}, vs Baseline Accuracy: {:.4f}'.format(acc, baseline_accuracy))
    return


if __name__ == "__main__":
    args = parser.parse_args()
    # Loading dataset

    dataset_builder = DatasetBuilder(args.dataset, only_binary=True, time_cutoff=1500)
    full_dataset = dataset_builder.create_dataset(dataset_type="sequential")
    val_dataset = full_dataset['val']
    val_dataset = [elt for elt in val_dataset if len(elt[0].shape) > 1]

    if args.debug:
        train_dataset = val_dataset
    else:
        train_dataset = full_dataset['train']
        # sometimes the sequence has only one element because of the time_cutoff decision
        train_dataset = [elt for elt in train_dataset if len(elt[0].shape) > 1]
    # import pdb;
    # pdb.set_trace()
    print("Some stats about the training data: {} data points, {} features".format(len(train_dataset),
                                                                                   train_dataset[0][0].size(1)))
    baseline_acc = sum([elt[1].item() for elt in train_dataset])/len(train_dataset)
    baseline_acc = max(1 - baseline_acc, baseline_acc)
    print(f"Baseline accuracy is {baseline_acc:.2f}")
    # Setting up model
    model = LSTMClassifier(input_size=train_dataset[0][0].size(1), h_size=args.hidden_size, n_layers=args.num_layers,
                           dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(args, model, optimizer, train_dataset, val_dataset, baseline_acc)

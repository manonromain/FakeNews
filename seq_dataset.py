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
parser.add_argument('--lr', default=0.05,
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
        _, (h_n, _) = self.lstm(seq)
        output = torch.cat([h_n[i, :, :] for i in range(self.n_layers)], dim=-1)
        return self.linear(output)


def dataset_iterator(dataset, batch_size=32, shuffle=True):
    if shuffle:
        random.shuffle(dataset)
    count_sampled = 0
    n = len(dataset)
    while (count_sampled < n):
        yield dataset[count_sampled:count_sampled + batch_size]
        count_sampled += batch_size


def train(args, model, optim, train_loader, test_loader=None):
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
        for ix, batch in enumerate(dataset_iterator(train_loader, batch_size=args.batch_size, shuffle=True)):
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
            print(f"Step {ix+1}, running loss: {running_loss:.4f}")

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
            for batch in dataset_iterator(train_loader, batch_size=args.batch_size, shuffle=True):
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
        print('Accuracy: {:.4f}'.format(acc))
        print('True_positives {} over {}'.format(correct, n_samples))
    return


if __name__ == "__main__":
    args = parser.parse_args()
    # Loading dataset

    dataset_builder = DatasetBuilder(args.dataset, only_binary=True, time_cutoff=1500)
    full_dataset = dataset_builder.create_dataset(dataset_type="sequential")
    train_dataset = full_dataset['train']
    train_dataset = [elt for elt in train_dataset if len(elt[0].shape) > 1]
    val_dataset = full_dataset['val']
    val_dataset = [elt for elt in val_dataset if len(elt[0].shape) > 1]
    # import pdb;
    # pdb.set_trace()
    print("Some stats about the training data: {} data points, {} features".format(len(train_dataset),
                                                                                   train_dataset[0][0].size(1)))
    # Setting up model
    model = LSTMClassifier(input_size=train_dataset[0][0].size(1), h_size=args.hidden_size, n_layers=args.num_layers,
                           dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(args, model, optimizer, train_dataset, val_dataset)

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset
from models import FirstNet, GNNStack
from dataset import DatasetBuilder
import numpy as np
import csv

def train(dataset, args):

    on_gpu = torch.cuda.is_available()
    if on_gpu:
        print("Using gpu")

    # Loading dataset

    time_cutoff = None if args.time_cutoff == "None" else int(args.time_cutoff)
    dataset_builder = DatasetBuilder(dataset, only_binary=args.only_binary, features_to_consider=args.features,
                                    time_cutoff=time_cutoff, seed=args.seed)
    datasets = dataset_builder.create_dataset(standardize_features=args.standardize, 
                                            on_gpu=on_gpu, oversampling_ratio=args.oversampling_ratio)
    train_data_loader = torch_geometric.data.DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_data_loader = torch_geometric.data.DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=True)
    test_data_loader = torch_geometric.data.DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=True)

    print("Number of node features", dataset_builder.num_node_features)
    print("Dimension of hidden space", args.hidden_dim)

    # Setting up model
    model = GNNStack(dataset_builder.num_node_features, args.hidden_dim, dataset_builder.num_classes, args)
    # model = GNNStack(dataset.num_node_features, 32, dataset.num_classes, args)
    if on_gpu:
        model.cuda()

    # Tensorboard logging
    log_dir = os.path.join("logs", args.exp_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    train_writer = SummaryWriter(os.path.join(log_dir, "train"))
    val_writer = SummaryWriter(os.path.join(log_dir, "val"))
    test_writer = SummaryWriter(os.path.join(log_dir, "test"))
    
    # CSV logging
    csv_logging = []
    
    # Checkpoints
    checkpoint_dir = os.path.join("checkpoints", args.exp_name)
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    if args.exp_name == "default" or not os.path.isfile(checkpoint_path):
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        epoch_ckp = 0
        global_step = 0
        best_val_acc = 0
    else:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch_ckp = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_val_acc = checkpoint["best_val_acc"]
        print("Restoring previous model at epoch", epoch_ckp)

    # Training phase
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    for epoch in range(epoch_ckp, epoch_ckp + args.num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_data_loader:
            # print(batch)
            # import pdb; pdb.set_trace()
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            epoch_loss += loss.sum().item()

            # Optimization
            loss.backward()
            optimizer.step()

            # TFBoard logging
            train_writer.add_scalar("loss", loss.mean(), global_step)
            global_step += 1

        print("epoch", epoch, "loss:", epoch_loss / len(train_data_loader))
        if epoch%1==0:
            # Evaluation on the training set 
            model.eval()
            correct = 0
            n_samples = 0
            samples_per_label = np.zeros(dataset_builder.num_classes)
            pred_per_label = np.zeros(dataset_builder.num_classes)
            correct_per_label = np.zeros(dataset_builder.num_classes)
            with torch.no_grad():
                for batch in train_data_loader:
                    _, pred = model(batch).max(dim=1)
                    correct += float(pred.eq(batch.y).sum().item())
                    for i in range(dataset_builder.num_classes):
                        batch_i = batch.y.eq(i)
                        pred_i = pred.eq(i)
                        samples_per_label[i] += batch_i.sum().item()
                        pred_per_label[i] += pred_i.sum().item()
                        correct_per_label[i] += (batch_i*pred_i).sum().item()
                    n_samples += len(batch.y)
            train_acc = correct / n_samples
            acc_per_label = correct_per_label / samples_per_label
            rec_per_label = correct_per_label / pred_per_label
            train_writer.add_scalar("Accuracy", train_acc, epoch)
            for i in range(dataset_builder.num_classes):
                train_writer.add_scalar("Accuracy_{}".format(i), acc_per_label[i], epoch)
                train_writer.add_scalar("Recall_{}".format(i), rec_per_label[i], epoch)
            print('Training accuracy: {:.4f}'.format(train_acc))

            # Evaluation on the validation set 
            model.eval()
            correct = 0
            n_samples = 0
            samples_per_label = np.zeros(dataset_builder.num_classes)
            pred_per_label = np.zeros(dataset_builder.num_classes)
            correct_per_label = np.zeros(dataset_builder.num_classes)
            with torch.no_grad():
                for batch in val_data_loader:
                    _, pred = model(batch).max(dim=1)
                    correct += float(pred.eq(batch.y).sum().item())
                    for i in range(dataset_builder.num_classes):
                        batch_i = batch.y.eq(i)
                        pred_i = pred.eq(i)
                        samples_per_label[i] += batch_i.sum().item()
                        pred_per_label[i] += pred_i.sum().item()
                        correct_per_label[i] += (batch_i*pred_i).sum().item()
                    n_samples += len(batch.y)
            val_acc = correct / n_samples
            acc_per_label = correct_per_label / samples_per_label
            rec_per_label = correct_per_label / pred_per_label
            val_writer.add_scalar("Accuracy", val_acc, epoch)
            for i in range(dataset_builder.num_classes):
                val_writer.add_scalar("Accuracy_{}".format(i), acc_per_label[i], epoch)
                val_writer.add_scalar("Recall_{}".format(i), rec_per_label[i], epoch)
            print('Validation accuracy: {:.4f}'.format(val_acc))
            
                 
            # Evaluation on the test set 
            model.eval()
            correct = 0
            n_samples = 0
            samples_per_label = np.zeros(dataset_builder.num_classes)
            pred_per_label = np.zeros(dataset_builder.num_classes)
            correct_per_label = np.zeros(dataset_builder.num_classes)
            with torch.no_grad():
                for batch in test_data_loader:
                    _, pred = model(batch).max(dim=1)
                    correct += float(pred.eq(batch.y).sum().item())
                    for i in range(dataset_builder.num_classes):
                        batch_i = batch.y.eq(i)
                        pred_i = pred.eq(i)
                        samples_per_label[i] += batch_i.sum().item()
                        pred_per_label[i] += pred_i.sum().item()
                        correct_per_label[i] += (batch_i*pred_i).sum().item()
                    n_samples += len(batch.y)
            test_acc = correct / n_samples
            acc_per_label = correct_per_label / samples_per_label
            rec_per_label = correct_per_label / pred_per_label
            test_writer.add_scalar("Accuracy", test_acc, epoch)
            for i in range(dataset_builder.num_classes):
                test_writer.add_scalar("Accuracy_{}".format(i), acc_per_label[i], epoch)
                test_writer.add_scalar("Recall_{}".format(i), rec_per_label[i], epoch)
            print('Test accuracy: {:.4f}'.format(test_acc))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Saving model if model is better
                checkpoint = {
                     "epoch": epoch,
                     "model_state_dict": model.state_dict(),
                     "epoch_loss": epoch_loss / len(train_data_loader),
                     "global_step": global_step,
                     "best_val_acc": best_val_acc
                }
                torch.save(checkpoint, checkpoint_path)
                
                dict_logging = vars(args).copy()
                dict_logging["train_acc"] = train_acc
                dict_logging["val_acc"] = val_acc
                dict_logging["test_acc"] = test_acc
                csv_logging.append(dict_logging)
    
    csv_exists = os.path.exists("results.csv")
    header = dict_logging.keys()

    with open("results.csv", "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        if not csv_exists:
            writer.writeheader()
        for dict_ in csv_logging:
            writer.writerow(dict_)
    return


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    parser = argparse.ArgumentParser(description='Train the graph network.')
    parser.add_argument('dataset', choices=["twitter15", "twitter16"],
                    help='Training dataset', default="twitter15")
    parser.add_argument('--lr', default=0.01, type=float,
                    help='learning rate')
    parser.add_argument('--num_epochs', default=200, type=int, 
                    help='Number of epochs')
    parser.add_argument('--oversampling_ratio', default=1, type=int, 
                    help='Oversampling ratio for data augmentation')
    parser.add_argument('--num_layers', default=2, type=int,
                    help='Number of layers')
    parser.add_argument('--dropout', default=0.0, type=float,
                    help='dropout for GNNStack')
    parser.add_argument('--model_type', default="GAT",
                    help='Model type for GNNStack')
    parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch_size')
    parser.add_argument('--only_binary', action='store_true',
                    help='Reduces the problem to binary classification')
    parser.add_argument('--exp_name', default="default",
                    help="Name of experiment - different names will log in different tfboards and restore different models")
    parser.add_argument('--standardize', action='store_true',
                    help='Standardize features')
    parser.add_argument('--features', choices=["all", "text_only", "user_only"],
                    help='Features to consider', default="all")
    parser.add_argument('--time_cutoff',
                    help='Time cutoff in mins', default="None")
    parser.add_argument('--seed', default=64, type=int,
                    help='Seed for train/val/test split')
    parser.add_argument('--hidden_dim', default=64, type=int,
                    help='Dimension of hidden space in GCNs')
   
    args = parser.parse_args()
    train(args.dataset, args)

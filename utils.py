import numpy as np
from datetime import datetime, timedelta
import os
import glob


def parse_edge_line(line):
    orig, dest = line.split("->")
    orig_list = orig.split("'")
    dest_list = dest.split("'")

    tweet_in, tweet_out = int(orig_list[3]), int(dest_list[3])
    user_in, user_out = int(orig_list[1]), int(dest_list[1])
    time_in, time_out = float(orig_list[5]), float(dest_list[5])
    return tweet_in, tweet_out, user_in, user_out, time_in, time_out


def get_root_id(tree_file_name):
    return int(os.path.splitext(os.path.basename(tree_file_name))[0])


def get_tree_file_names(datadir):
    return glob.glob(os.path.join(datadir, "tree", "*.txt"))


def one_hot_label(label):
    if label == "non-rumor":
        return np.array([[1, 0, 0, 0]])
    if label == "false":
        return np.array([[0, 1, 0, 0]])
    if label == "true":
        return np.array([[0, 0, 1, 0]])
    if label == "unverified":
        return np.array([[0, 0, 0, 1]])


def to_label(label):
    if label == "false":
        return np.array([0])
    if label == "true":
        return np.array([1])
    if label == "non-rumor":
        return np.array([2])
    if label == "unverified":
        return np.array([3])


def from_date_text_to_timestamp(datestr):
    year, month, day = map(int, datestr.split()[0].split("-"))
    return (datetime(year, month, day) - datetime(1970, 1, 1)) / timedelta(days=1)


def cap_sequences(sequential_dataset, cap_len):
    sequential_dataset = [elt for elt in sequential_dataset if len(elt[0].shape) > 1]
    sequential_dataset = [[elt[0][:cap_len, :], elt[1]] for elt in sequential_dataset if elt[0].size(0) >= cap_len]
    return sequential_dataset

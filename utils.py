import numpy as np


def parse_edge_line(line):
    orig, dest = line.split("->")
    orig_list = orig.split("'")
    dest_list = dest.split("'")

    tweet_in, tweet_out = int(orig_list[3]), int(dest_list[3])
    user_in, user_out = int(orig_list[1]), int(dest_list[1])
    time_in, time_out = float(orig_list[5]), float(dest_list[5])
    return tweet_in, tweet_out, user_in, user_out, time_in, time_out


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


def from_date_text_to_seconds(datestr):
    return NotImplementedError
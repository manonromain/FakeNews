import glob
import os
import time

import numpy as np
import torch
import torch_geometric

from text_preprocessing import preprocess_tweets

DATA_DIR = "rumor_detection_acl2017"


def parse_edge_line(line):
    orig, dest = line.split("->")
    orig_list = orig.split("'")
    dest_list = dest.split("'")

    orig = [int(orig_list[3]), float(orig_list[5])]
    dest = [int(dest_list[3]), float(dest_list[5])]
    # print(orig, dest)
    return orig, dest


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
    if label == "non-rumor":
        return np.array([0])
    if label == "false":
        return np.array([1])
    if label == "true":
        return np.array([2])
    if label == "unverified":
        return np.array([3])


def data_twitter(dataset="twitter15"):
    starttime = time.time()

    dataset_dir = os.path.join(DATA_DIR, dataset)
    if not os.path.isdir(dataset_dir):
        print("Dataset dir doesn't exist")
        return

    labels = {}
    with open(os.path.join(dataset_dir, "label.txt")) as label_file:
        for line in label_file.readlines():
            label, tweet_id = line.split(":")
            labels[int(tweet_id)] = label

    tweet_dict = {}
    with open(os.path.join(dataset_dir, "source_tweets.txt")) as text_file:
        for line in text_file.readlines():
            tweet_id, text = line.split("\t")
            try:
                tweet_dict[int(tweet_id)] = text
            except NotImplementedError:
                print("Text features will not be used")
                break

    text_features = preprocess_tweets(tweet_dict)
    print("Tweets tf-idfed in {:3f}s".format(time.time() - starttime))

    trees_to_parse = glob.glob(os.path.join(dataset_dir, "tree", "*.txt"))

    dataset = []

    #
    n_text_features = (text_features[int(tweet_id)].shape[0] if text_features else 0)  # FIXME + text_features
    for tree_file_name in trees_to_parse:
        edges = []
        x = []
        tweet_id_to_uid = {}  # TODO could create a specific data structure
        count = 0

        root_tweet_id = int(os.path.splitext(os.path.basename(tree_file_name))[0])
        label = labels[root_tweet_id]
        # print("root_tweet_id", root_tweet_id)
        with open(tree_file_name, "rt") as tree_file:
            for line in tree_file.readlines():
                if "ROOT" in line:
                    continue
                orig, dest = parse_edge_line(line)

                # Add orig if unseen
                if orig[0] not in tweet_id_to_uid:
                    tweet_id_to_uid[orig[0]] = count
                    if orig[0] in text_features:
                        text_ft = text_features[orig[0]]
                    else:
                        text_ft = np.zeros(n_text_features)
                    features = np.append(text_ft, orig[1])
                    x.append(features)

                    count += 1
                # Add dest if unseen
                if dest[0] not in tweet_id_to_uid:
                    tweet_id_to_uid[dest[0]] = count
                    if dest[0] in text_features:
                        text_ft = text_features[dest[0]]
                    else:
                        text_ft = np.zeros(n_text_features)
                    features = np.append(text_ft, dest[1])
                    x.append(features)

                    count += 1

                # Add edge
                edges.append(np.array([tweet_id_to_uid[orig[0]], tweet_id_to_uid[dest[0]]]))

        y = torch.tensor(to_label(label))
        edge_index = torch.tensor(np.array(edges)).t().contiguous()
        x = torch.tensor(x, dtype=torch.float32)
        # print(x.shape, edge_index.shape)
        # print(x)
        # print("Number of nodes", count)
        dataset.append(torch_geometric.data.Data(x=x, y=y, edge_index=edge_index))
    print("Dataset loaded in ", time.time() - starttime, "s")
    return dataset


if __name__ == "__main__":
    data_twitter("twitter15")

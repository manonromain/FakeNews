import glob
import os
import time

import numpy as np
import torch
import torch_geometric

from text_preprocessing import preprocess_tweets
from utils import to_label, parse_edge_line

import random

DATA_DIR = "rumor_detection_acl2017"


class DatasetBuilder:

    def __init__(self, dataset="twitter15"):
        self.dataset = dataset

        self.dataset_dir = os.path.join(DATA_DIR, dataset)
        if not os.path.isdir(self.dataset_dir):
            raise IOError(f"{self.dataset_dir} doesn't exist")

    def load_labels(self):
        """
        Returns:
            labels: dict[tweet_id:int -> label:int]
        """

        labels = {}
        with open(os.path.join(self.dataset_dir, "label.txt")) as label_file:
            for line in label_file.readlines():
                label, tweet_id = line.split(":")
                labels[int(tweet_id)] = label
        return labels

    def load_tweet_texts(self):
        """
        Returns:
            labels: dict[tweet_id:int -> text:str]
        """

        tweet_texts = {}

        # with open(os.path.join(DATA_DIR, "tweet_features.txt")) as text_file:
        #     text_file.readline() #first line contains column names 
        #     for line in text_file.readlines():
        #         tweet_id, text, _ = line.split(";")
        #         tweet_texts[int(tweet_id)] = text

        with open(os.path.join(self.dataset_dir, "source_tweets.txt")) as text_file:
            for line in text_file.readlines():
                tweet_id, text = line.split("\t")
                tweet_texts[int(tweet_id)] = text
    
        return tweet_texts

    def create_dataset(self):
        starttime = time.time()

        labels = self.load_labels()
        tweet_texts = self.load_tweet_texts()

        text_features = preprocess_tweets(tweet_texts)
        print("Tweets tf-idfed in {:3f}s".format(time.time() - starttime))

        trees_to_parse = glob.glob(os.path.join(self.dataset_dir, "tree", "*.txt"))

        dataset = []

        n_text_features = len(list(text_features.values())[0])
        # n_text_features = (text_features[int(tweet_id)].shape[0] if text_features else 0)  # FIXME + text_features
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
                    tweet_in, tweet_out, user_in, time_in, time_out = parse_edge_line(line)

                    # Add orig if unseen
                    if tweet_in not in tweet_id_to_uid:
                        tweet_id_to_uid[tweet_in] = count
                        if tweet_in in text_features:
                            text_ft = text_features[tweet_in]
                        else:
                            text_ft = np.zeros(n_text_features)
                        features = np.append(text_ft, time_in)
                        x.append(features)

                        count += 1
                    # Add dest if unseen
                    if tweet_out not in tweet_id_to_uid:
                        tweet_id_to_uid[tweet_out] = count
                        if tweet_out in text_features:
                            text_ft = text_features[tweet_out]
                        else:
                            text_ft = np.zeros(n_text_features)
                        features = np.append(text_ft, time_out)
                        x.append(features)

                        count += 1

                    # Add edge
                    edges.append(np.array([tweet_id_to_uid[tweet_in], tweet_id_to_uid[tweet_out]]))

            y = torch.tensor(to_label(label))
            edge_index = torch.tensor(np.array(edges)).t().contiguous()  # Why?
            x = torch.tensor(x, dtype=torch.float32)
            # print(x.shape, edge_index.shape)
            # print(x)
            # print("Number of nodes", count)
            dataset.append(torch_geometric.data.Data(x=x, y=y, edge_index=edge_index))
        print("Dataset loaded in ", time.time() - starttime, "s")
        return dataset

if __name__ == "__main__":
    data_builder = DatasetBuilder("twitter15")
    data_builder.create_dataset()

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

    def __init__(self, dataset="twitter15", only_binary=True):
        self.dataset = dataset

        self.dataset_dir = os.path.join(DATA_DIR, dataset)
        if not os.path.isdir(self.dataset_dir):
            raise IOError(f"{self.dataset_dir} doesn't exist")

        self.only_binary = only_binary
        if self.only_binary:
            print("Considering only binary classification problem")
        else:
            print("Considering 4 classes problem")


    def create_dataset(self):
        start_time = time.time()

        self.labels = self.load_labels()
        self.tweet_texts = self.load_tweet_texts()

        self.load_user_features()

        # TODO: ADD PREPROCESSING ON TWEET_TEXTS, FOR NOW THEY ARE SUPER UGLY

        print('running tf-idf')
        self.text_features = preprocess_tweets(self.tweet_texts)
        self.n_text_features = len(list(self.text_features.values())[0])
        print("Tweets tf-idfed in {:3f}s".format(time.time() - start_time))

        trees_to_parse = glob.glob(os.path.join(self.dataset_dir, "tree", "*.txt"))

        dataset = []

        for tree_file_name in trees_to_parse:

            root_tweet_id = int(os.path.splitext(os.path.basename(tree_file_name))[0])
            label = self.labels[root_tweet_id]

            if (not self.only_binary) or (label in ['false', 'true']):
                dataset.append(self.build_tree(tree_file_name))

        print(f"Dataset loaded in {time.time() - start_time}s")

        return dataset


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
            tweet_texts: dict[tweet_id:int -> text:str]
        """

        tweet_texts = {}

        with open(os.path.join(DATA_DIR, "tweet_features.txt")) as text_file:
            text_file.readline() #first line contains column names 
            for line in text_file.readlines():
                tweet_id, text, _ = line.split(";")
                tweet_texts[int(tweet_id)] = text

        with open(os.path.join(self.dataset_dir, "source_tweets.txt")) as text_file:
            for line in text_file.readlines():
                tweet_id, text = line.split("\t")
                tweet_texts[int(tweet_id)] = text

        return tweet_texts


    def load_user_features(self):
        """
        Returns:
            user_features: dict[tweet_id:int -> dict[name feature -> feature]]
        """
        user_features = {}

        with open(os.path.join(DATA_DIR, "user_features.txt")) as text_file:
            feature_names = text_file.readline().rstrip('\n').split(';') #first line contains column names 
            for line in text_file.readlines():
                features = line.rstrip('\n').split(";")
                user_features[features[0]] = {feature_names[i]:features[i] 
                                              for i in range(1, len(features))}
        return user_features


    def build_tree(self, tree_file_name):
        """ Parses the file to build a tree, adding all the features.

        Args:
            tree_file_name:str (path to the file storing the tree)

        Returns:
            torch_geometric.data.Data, which contains
                (x:Tensor(n_nodes * n_features)
                 y:Tensor(n_nodes)
                 edge_index: Tensor(2 * E))
        """

        edges = [] #
        x = []
        tweet_id_to_count = {} # Dict tweet id -> node id, which starts at 0
        count = 0

        root_tweet_id = int(os.path.splitext(os.path.basename(tree_file_name))[0])
        label = self.labels[root_tweet_id]

        with open(tree_file_name, "rt") as tree_file:
            for line in tree_file.readlines():
                if "ROOT" in line:
                    continue
                tweet_in, tweet_out, user_in, user_out, time_in, time_out = parse_edge_line(line)

                # Add orig if unseen
                if tweet_in not in tweet_id_to_count:
                    tweet_id_to_count[tweet_in] = count
                    text_features = (np.zeros(self.n_text_features) 
                                if tweet_in not in self.text_features
                                else self.text_features[tweet_in])
                    features = np.append(text_features, time_in)
                    x.append(features)
                    count += 1
                
                # Add dest if unseen
                if tweet_out not in tweet_id_to_count:
                    tweet_id_to_count[tweet_out] = count
                    text_features = (np.zeros(self.n_text_features) 
                                if tweet_out not in self.text_features
                                else self.text_features[tweet_out])
                    features = np.append(text_features, time_out)
                    x.append(features)
                    count += 1

                # Add edge
                edges.append(np.array([tweet_id_to_count[tweet_in], 
                                      tweet_id_to_count[tweet_out]]))

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(to_label(label))
        edge_index = torch.tensor(np.array(edges)).t().contiguous() 
        
        return torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)

if __name__ == "__main__":
    data_builder = DatasetBuilder("twitter15")
    data_builder.create_dataset()

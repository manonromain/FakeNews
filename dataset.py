import glob
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch_geometric

import utils
from text_preprocessing import preprocess_tweets

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

        labels = self.load_labels()

        tweet_features = self.load_tweet_features()
        preprocessed_tweet_fts = self.preprocess_tweet_features(tweet_features)

        user_features = self.load_user_features()
        preprocessed_user_fts = self.preprocess_user_features(user_features)

        trees_to_parse = glob.glob(os.path.join(self.dataset_dir, "tree", "*.txt"))

        dataset = []

        for tree_file_name in trees_to_parse:

            root_tweet_id = int(os.path.splitext(os.path.basename(tree_file_name))[0])
            label = labels[root_tweet_id]

            if (not self.only_binary) or (label in ['false', 'true']):
                dataset.append(self.build_tree(tree_file_name, preprocessed_tweet_fts, preprocessed_user_fts, labels))

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

    def load_tweet_features(self):
        """
        Returns:
            tweet_texts: dict[tweet_id:int -> dict[name feature -> feature]]
        """

        tweet_features = {}

        with open(os.path.join(DATA_DIR, "tweet_features.txt")) as text_file:
            #first line contains column names 
            self.tweet_feature_names = text_file.readline().rstrip('\n').split(';') 
            for line in text_file.readlines():
                features = line.rstrip('\n').split(";")
                tweet_features[int(features[0])] = {self.tweet_feature_names[i]:features[i] 
                                              for i in range(1, len(features))}

        with open(os.path.join(self.dataset_dir, "source_tweets.txt")) as text_file:
            for line in text_file.readlines():
                tweet_id, text = line.split("\t")
                if tweet_id not in tweet_features.keys():
                    tweet_features[int(tweet_id)] = {'text':text, 
                                                'created_at':'2016-01-01 00:00:01'}     
                    # TODO: change the date according to if dataset is 2015 or 2016

        return tweet_features


    def load_user_features(self):
        """
        Returns:
            user_features: dict[tweet_id:int -> dict[name feature -> feature]]
        """
        user_features = {}

        with open(os.path.join(DATA_DIR, "user_features.txt")) as text_file:
            #first line contains column names 
            self.user_feature_names = text_file.readline().rstrip('\n').split(';') 
            for line in text_file.readlines():
                features = line.rstrip('\n').split(";")
                user_features[int(features[0])] = {self.user_feature_names[i]:features[i] 
                                              for i in range(1, len(features))}
        return user_features


    def preprocess_tweet_features(self, tweet_features):
        """ Preprocess all tweet features to transform dicts into fixed-sized array.

        Args:
            tweet_features: dict[tweet_id -> dict[name_feature -> feature]]
        Returns:
            defaultdict[tweet_id -> np.array(n_dim)]

        """

        #TODO: more preprocessing, this is just a beginning.
        if 'created_at' in self.tweet_feature_names:
            for tweet_id in tweet_features.keys():
                tweet_features[tweet_id]['created_at'] = \
                    utils.from_date_text_to_timestamp(tweet_features[tweet_id]['created_at'])

        # print('running tf-idf')
        # self.text_features = preprocess_tweets(self.tweet_texts)
        # self.n_text_features = len(list(self.text_features.values())[0])
        # print("Tweets tf-idfed in {:3f}s".format(time.time() - start_time))

        def default_tweet_features():
            return np.array([utils.from_date_text_to_timestamp('2016-01-01 00:00:01')])
        
        new_tweet_features = {key:np.array([val['created_at']]) for key, val in tweet_features.items()}
        return defaultdict(default_tweet_features, new_tweet_features)

    def preprocess_user_features(self, user_features):
        """ Preprocess all user features to transform dicts into fixed-sized array.

        Args:
            user_features: dict[user_id -> dict[name_feature -> feature]]
        Returns:
            defaultdict[user_id -> np.array(n_dim)]
        
        """
        
        #TODO: more preprocessing, this is just a beginning.
        if 'created_at' in self.user_feature_names:
            for user_id in user_features.keys():
                user_features[user_id]['created_at'] = \
                    utils.from_date_text_to_timestamp(user_features[user_id]['created_at'])

        def default_user_features():
            return np.array([utils.from_date_text_to_timestamp('2016-01-01 00:00:01')])
        
        new_user_features = {key:np.array([val['created_at']]) for key, val in user_features.items()}
        return defaultdict(default_user_features, new_user_features)

    def build_tree(self, tree_file_name, tweet_fts, user_fts, labels):
        """ Parses the file to build a tree, adding all the features.

        Args:
            tree_file_name:str (path to the file storing the tree)
            tweet_fts: dict[tweet_id:int -> tweet-features:np array]
            user_fts: dict[user_id:int -> user-features:np array]
            labels: dict[tweet_id:int -> label:int]

        Returns:
            torch_geometric.data.Data, which contains
                (x:Tensor(n_nodes * n_features)
                 y:Tensor(n_nodes)
                 edge_index: Tensor(2 * E))
        """

        def agglomerate_features(node_tweet_fts, node_user_fts):
            return np.concatenate([node_tweet_fts, node_user_fts])

            

        edges = [] #
        x = []
        tweet_id_to_count = {} # Dict tweet id -> node id, which starts at 0
        count = 0

        root_tweet_id = int(os.path.splitext(os.path.basename(tree_file_name))[0])
        label = labels[root_tweet_id]

        self.number_of_features = len(agglomerate_features(tweet_fts[-1], user_fts[-1]))

        with open(tree_file_name, "rt") as tree_file:
            for line in tree_file.readlines():
                if "ROOT" in line:
                    continue
                tweet_in, tweet_out, user_in, user_out, time_in, time_out = utils.parse_edge_line(line)

                # Add orig if unseen
                if tweet_in not in tweet_id_to_count:
                    tweet_id_to_count[tweet_in] = count
                    features_node = agglomerate_features(tweet_fts[tweet_in], user_fts[user_in])
                    x.append(features_node)
                    count += 1
                
                # Add dest if unseen
                if tweet_out not in tweet_id_to_count:
                    tweet_id_to_count[tweet_out] = count
                    features_node = agglomerate_features(tweet_fts[tweet_out], user_fts[user_out])
                    x.append(features_node)
                    count += 1

                # Add edge
                edges.append(np.array([tweet_id_to_count[tweet_in], 
                                      tweet_id_to_count[tweet_out]]))

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(utils.to_label(label))
        edge_index = torch.tensor(np.array(edges)).t().contiguous() 
        
        return torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)


if __name__ == "__main__":
    data_builder = DatasetBuilder("twitter15")
    data_builder.create_dataset()

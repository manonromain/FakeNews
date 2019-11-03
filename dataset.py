import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch_geometric

import utils
from text_preprocessing import preprocess_tweets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = "rumor_detection_acl2017"


class DatasetBuilder:

    def __init__(self, dataset="twitter15", only_binary=True, time_cutoff=None):

        self.dataset = dataset

        self.dataset_dir = os.path.join(DATA_DIR, dataset)
        if not os.path.isdir(self.dataset_dir):
            raise IOError(f"{self.dataset_dir} doesn't exist")

        self.only_binary = only_binary
        if self.only_binary:
            print("Considering only binary classification problem")
        else:
            print("Considering 4 classes problem")

        self.time_cut = time_cutoff
        if self.time_cut is not None:
            print("We consider tweets emitted no later than {}mins after the root tweet".format(self.time_cut))
        else:
            print("No time consideration")

    def create_dataset(self, dataset_type="graph"):
        """
        Args:
            dataset_type:str. Has to be "graph", "sequential" or "raw"
        Returns:
            dict with keys "train", "val", "test":
                If dataset_type is "graph" contains list of
                    torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)
                If dataset_type is "sequential" contains list of
                    (sequential_data, y)
        """
        if dataset_type not in ["graph", "sequential", "raw"]:
            raise ValueError("Supported dataset types are: 'graph', 'sequential', 'raw'.")

        start_time = time.time()

        trees_to_parse = utils.get_tree_file_names(self.dataset_dir)

        labels = self.load_labels()

        # Create train-val-test split
        # Remove useless trees (i.e. with labels that we don't consider)

        news_ids_to_consider = list(labels.keys())
        if self.only_binary:
            news_ids_to_consider = [news_id for news_id in news_ids_to_consider
                                    if labels[news_id] in ['false', 'true']]

        train_ids, val_ids = train_test_split(news_ids_to_consider, test_size=0.1, random_state=42)
        train_ids, test_ids = train_test_split(train_ids, test_size=0.25, random_state=68)
        print(f"Len train/val/test {len(train_ids)} {len(val_ids)} {len(test_ids)}")

        user_ids_in_train, tweet_ids_in_train = \
            self.get_user_and_tweet_ids_in_train(trees_to_parse, train_ids)

        tweet_features = self.load_tweet_features()
        user_features = self.load_user_features()

        preprocessed_tweet_fts = self.preprocess_tweet_features(tweet_features, tweet_ids_in_train)
        preprocessed_user_fts = self.preprocess_user_features(user_features, user_ids_in_train)

        ids_to_dataset = {news_id: 'train' for news_id in train_ids}
        ids_to_dataset.update({news_id: 'val' for news_id in val_ids})
        ids_to_dataset.update({news_id: 'test' for news_id in test_ids})

        dataset = {'train': [], 'val': [], 'test': []}

        for tree_file_name in trees_to_parse:

            news_id = utils.get_root_id(tree_file_name)
            label = labels[news_id]

            if (not self.only_binary) or (label in ['false', 'true']):

                node_features, edges = self.build_tree(tree_file_name, tweet_fts=preprocessed_tweet_fts,
                                                       user_fts=preprocessed_user_fts)

                if dataset_type == "graph":
                    x = torch.tensor(node_features, dtype=torch.float32)
                    y = torch.tensor(utils.to_label(label))
                    edge_index = np.array([edge[:2] for edge in edges],
                                          dtype=int)  # change if you want the time somewhere
                    edge_index = torch.tensor(edge_index).t().contiguous()
                    dataset[ids_to_dataset[news_id]].append(torch_geometric.data.Data(x=x, y=y, edge_index=edge_index))

                elif dataset_type == "sequential":
                    y = torch.tensor(utils.to_label(label))
                    ordered_edges = sorted(edges, key=lambda x: x[2])
                    sequential_data = torch.tensor([node_features[edge[1]] for edge in ordered_edges])
                    dataset[ids_to_dataset[news_id]].append([sequential_data, y])
                    # print(sequential_data.mean(dim=0))
                    # print("label was {}".format(label))
                elif dataset_type == "raw":
                    dataset[ids_to_dataset[news_id]].append(
                        [[label, news_id] + edge + list(node_features[edge[1]]) for edge in edges])

        print(f"Dataset loaded in {time.time() - start_time:.3f}s")

        return dataset

    def load_labels(self):
        """
        Returns:
            labels: dict[news_id:int -> label:int]
        """
        labels = {}
        with open(os.path.join(self.dataset_dir, "label.txt")) as label_file:
            for line in label_file.readlines():
                label, news_id = line.split(":")
                labels[int(news_id)] = label
        return labels

    def load_tweet_features(self):
        """
        Returns:
            tweet_texts: dict[tweet_id:int -> dict[name feature -> feature]]
        """

        tweet_features = {}

        with open(os.path.join(DATA_DIR, "tweet_features.txt")) as text_file:
            # first line contains column names
            self.tweet_feature_names = text_file.readline().rstrip('\n').split(';')
            for line in text_file.readlines():
                features = line.rstrip('\n').split(";")
                tweet_features[int(features[0])] = {self.tweet_feature_names[i]: features[i]
                                                    for i in range(1, len(features))}

        with open(os.path.join(self.dataset_dir, "source_tweets.txt")) as text_file:
            for line in text_file.readlines():
                tweet_id, text = line.split("\t")
                if tweet_id not in tweet_features.keys():
                    tweet_features[int(tweet_id)] = {'text': text,
                                                     'created_at': '2016-01-01 00:00:01'}
                    # TODO: change the date according to if dataset is 2015 or 2016

        return tweet_features

    def load_user_features(self):
        """
        Returns:
            user_features: dict[tweet_id:int -> dict[name feature -> feature]]
        """
        user_features = {}

        with open(os.path.join(DATA_DIR, "user_features.txt")) as text_file:
            # first line contains column names
            self.user_feature_names = text_file.readline().rstrip('\n').split(';')
            for line in text_file.readlines():
                features = line.rstrip('\n').split(";")
                user_features[int(features[0])] = {self.user_feature_names[i]: features[i]
                                                   for i in range(1, len(features))}
        return user_features

    def preprocess_tweet_features(self, tweet_features, tweet_ids_in_train):
        """ Preprocess all tweet features to transform dicts into fixed-sized array.

        Args:
            tweet_features: dict[tweet_id -> dict[name_feature -> feature]]
        Returns:
            defaultdict[tweet_id -> np.array(n_dim)]

        """

        # TODO: more preprocessing, this is just a beginning.
        if 'created_at' in self.tweet_feature_names:
            for tweet_id in tweet_features.keys():
                tweet_features[tweet_id]['created_at'] = \
                    utils.from_date_text_to_timestamp(tweet_features[tweet_id]['created_at'])

        # print('running tf-idf')
        # self.text_features = preprocess_tweets(self.tweet_texts)
        # self.n_text_features = len(list(self.text_features.values())[0])
        # print("Tweets tf-idfed in {:3f}s".format(time.time() - start_time))

        def default_tweet_features():
            # return np.array([utils.from_date_text_to_timestamp('2016-01-01 00:00:01')])
            return np.array([])

        # new_tweet_features = {key: np.array([val['created_at']]) for key, val in tweet_features.items()}
        new_tweet_features = {key: np.array([]) for key, val in tweet_features.items()}
        return defaultdict(default_tweet_features, new_tweet_features)

    def preprocess_user_features(self, user_features, user_ids_in_train):
        """ Preprocess all user features to transform dicts into fixed-sized array.

        Args:
            user_features: dict[user_id -> dict[name_feature -> feature]]
        Returns:
            defaultdict[user_id -> np.array(n_dim)]
        """

        # Available variables
        # id;
        # created_at;
        # description;
        # favourites_count;
        # followers_count;
        # friends_count;
        # geo_enabled;
        # listed_count;
        # location;
        # name;
        # screen_name;
        # statuses_count;
        # verified

        # Features we use:
        # created_at
        # favourites_count 
        # followers_count 
        # friends_count 
        # geo_enabled
        # has_description
        # len_name
        # len_screen_name
        # listed_count
        # statuses_count 
        # verified

        for user_id, features in user_features.items():

            new_features = {}  # will contain the processed features of current user

            if "created_at" in features:
                new_features['created_at'] = \
                    utils.from_date_text_to_timestamp(features['created_at'])

            integer_features = [
                "favourites_count",
                "followers_count",
                "friends_count",
                "listed_count",
                "statuses_count",
            ]
            #print(features.keys())
            for int_feature in integer_features:
                new_features[int_feature] = int(features[int_feature])

            new_features["verified"] = int(features['verified']=='True')
            new_features["geo_enabled"] = int(features['geo_enabled']=='True')
            new_features['has_description'] = int(len(features['description']) > 0)
            new_features['len_name'] = len(features['name'])
            new_features['len_screen_name'] = len(features['screen_name'])

            user_features[user_id] = new_features

        user_features_train_only = {key: val for key, val in user_features.items() if key in user_ids_in_train}

        # Standardizing
        for ft in [
            "created_at",
            "favourites_count",
            "followers_count",
            "friends_count",
            "listed_count",
            "statuses_count",
            "len_name",
            "len_screen_name"
        ]:
            scaler = StandardScaler().fit(
                np.array([val[ft] for val in user_features_train_only.values()]).reshape(-1, 1)
            )

            # faster to do this way as we don't have to convert to np arrays
            mean, std = scaler.mean_[0], scaler.var_[0] ** (1 / 2)
            for key in user_features_train_only.keys():
                user_features_train_only[key][ft] = (user_features_train_only[key][ft] - mean) / std

            for key in user_features.keys():
                user_features[key][ft] = (user_features[key][ft] - mean) / std

        dict_defaults = {
            'created_at': np.median([elt["created_at"] for elt in user_features_train_only.values()]),
            'favourites_count': np.median([elt["favourites_count"] for elt in user_features_train_only.values()]),
            'followers_count': np.median([elt["followers_count"] for elt in user_features_train_only.values()]),
            'friends_count': np.median([elt["friends_count"] for elt in user_features_train_only.values()]),
            'geo_enabled': 0,
            'has_description': 0,
            'len_name': np.median([elt["len_name"] for elt in user_features_train_only.values()]),
            'len_screen_name': np.median([elt["len_screen_name"] for elt in user_features_train_only.values()]),
            'listed_count': np.median([elt["listed_count"] for elt in user_features_train_only.values()]),
            'statuses_count': np.median([elt["statuses_count"] for elt in user_features_train_only.values()]),
            'verified': 0
        }

        def default_user_features():
            """ Return np array of default features sorted by alphabetic order """
            return np.array([val for key, val in
                             sorted(dict_defaults.items(), key=lambda x: x[0])])

        #  user features: key=uid, value=dict[ftname:valueft]
        np_user_features = {key: np.array([key_val[1] for key_val in sorted(value.items(), key=lambda x: x[0])]) for
                            key, value in user_features.items()}

        return defaultdict(default_user_features, np_user_features)

    def get_user_and_tweet_ids_in_train(self, trees_to_parse, train_ids):
        """ Returns sets of all the user ids and tweet ids that appear in train set """

        user_ids_in_train = set()
        tweet_ids_in_train = set()
        for tree_file_name in trees_to_parse:
            news_id = utils.get_root_id(tree_file_name)
            if news_id in train_ids:
                with open(tree_file_name, "rt") as tree_file:
                    for line in tree_file.readlines():
                        if "ROOT" in line:
                            continue
                        tweet_in, tweet_out, user_in, user_out, _, _ = utils.parse_edge_line(line)
                        user_ids_in_train.add(user_in)
                        user_ids_in_train.add(user_out)
                        tweet_ids_in_train.add(tweet_in)
                        tweet_ids_in_train.add(tweet_out)
        return user_ids_in_train, tweet_ids_in_train

    def build_tree(self, tree_file_name, tweet_fts, user_fts):
        """ Parses the file to build a tree, adding all the features.

        Args:
            tree_file_name:str (path to the file storing the tree)
            tweet_fts: dict[tweet_id:int -> tweet-features:np array]
            user_fts: dict[user_id:int -> user-features:np array]
            labels: dict[tweet_id:int -> label:int]

        Returns:
            x: list (n_nodes)[np.array (n_features)]
            edge_index: list (nb_edges)[node_in_id, node_out_id, time_out]
        """

        def agglomerate_features(node_tweet_fts, node_user_fts):
            return np.concatenate([node_tweet_fts, node_user_fts])

        edges = []  #
        x = []
        tweet_id_to_count = {}  # Dict tweet id -> node id, which starts at 0
        count = 0

        self.number_of_features = len(agglomerate_features(tweet_fts[-1], user_fts[-1]))

        with open(tree_file_name, "rt") as tree_file:
            for line in tree_file.readlines():
                if "ROOT" in line:
                    continue

                tweet_in, tweet_out, user_in, user_out, time_in, time_out = utils.parse_edge_line(line)

                if (self.time_cut is None) or (time_out >= 0 and time_out <= self.time_cut):
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
                    edges.append([tweet_id_to_count[tweet_in],
                                  tweet_id_to_count[tweet_out],
                                  time_out,
                                  user_in,
                                  user_out])

        return x, edges


if __name__ == "__main__":
    data_builder = DatasetBuilder("twitter15", time_cutoff=2000)
    dataset = data_builder.create_dataset(dataset_type="sequential")
    # import pdb;
    # pdb.set_trace()
    data_builder = DatasetBuilder("twitter16")
    data_builder.create_dataset(dataset_type="graph")

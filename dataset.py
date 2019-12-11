import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

import utils
from text_preprocessing import preprocess_tweets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = "rumor_detection_acl2017"

import basic_tests


class DatasetBuilder:

    def __init__(
        self, 
        dataset="twitter15", 
        only_binary=False, 
        time_cutoff=None, 
        features_to_consider="user_only", 
        seed=64
    ):
        
        self.seed = seed
        self.dataset = dataset

        self.dataset_dir = os.path.join(DATA_DIR, dataset)
        if not os.path.isdir(self.dataset_dir):
            raise IOError(f"{self.dataset_dir} doesn't exist")

        self.only_binary = only_binary
        if self.only_binary:
            self.num_classes = 2
            print("Considering only binary classification problem")
        else:
            self.num_classes = 4
            print("Considering 4 classes problem")

        self.time_cut = time_cutoff
        if self.time_cut is not None:
            print("We consider tweets emitted no later than {}mins after the root tweet".format(self.time_cut))
        else:
            print("No time consideration")

        print("Features that will be considered:", features_to_consider)
        self.features_to_consider = features_to_consider

    def create_dataset(self, dataset_type="graph", standardize_features=True, on_gpu=False, oversampling_ratio=1):
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

        train_ids, val_ids = train_test_split(news_ids_to_consider, test_size=0.1, random_state=self.seed)
        train_ids, test_ids = train_test_split(train_ids, test_size=0.25, random_state=self.seed*7)
        print(f"Len train/val/test {len(train_ids)} {len(val_ids)} {len(test_ids)}")

        user_ids_in_train, tweet_ids_in_train = \
            self.get_user_and_tweet_ids_in_train(trees_to_parse, train_ids)

        tweet_features = self.load_tweet_features()
        user_features = self.load_user_features()

        if standardize_features:
            print("Standardizing features")

        preprocessed_tweet_fts = self.preprocess_tweet_features(tweet_features, tweet_ids_in_train)
        preprocessed_user_fts = self.preprocess_user_features(user_features, user_ids_in_train, standardize_features)

        # basic_tests.test_user_preprocessed_features(preprocessed_user_fts)

        ids_to_dataset = {news_id: 'train' for news_id in train_ids}
        ids_to_dataset.update({news_id: 'val' for news_id in val_ids})
        ids_to_dataset.update({news_id: 'test' for news_id in test_ids})

        dataset = {'train': [], 'val': [], 'test': []}

        trees = []

        for tree_file_name in trees_to_parse:
            news_id = utils.get_root_id(tree_file_name)
            label = labels[news_id]
            if (not self.only_binary) or (label in ['false', 'true']):
                node_features, edges = self.build_tree(tree_file_name, tweet_fts=preprocessed_tweet_fts,
                                                       user_fts=preprocessed_user_fts)
                trees.append((news_id, label, node_features, edges))

        self.oversample(trees, ids_to_dataset, ratio=oversampling_ratio)

        for news_id, label, node_features, edges in trees:

            if dataset_type == "graph":
                import torch_geometric
                x = torch.tensor(node_features, dtype=torch.float32)
                y = torch.tensor(utils.to_label(label))
                edge_index = np.array([edge[:2] for edge in edges],
                                      dtype=int)  # change if you want the time somewhere
                edge_index = torch.tensor(edge_index).t().contiguous()
                if on_gpu:
                    y.to(torch.device("cuda"))
                    x.to(torch.device("cuda"))
                    edge_index.to(torch.device("cuda"))
                data_point = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index)
                if on_gpu:
                    data_point.to(torch.device("cuda"))
                dataset[ids_to_dataset[news_id]].append(data_point)

                # Uncomment for test, to see if graphs are well created
                # if news_id in [580320684305416192, 387021726007042051]:
                #     basic_tests.inspect_graph(dataset[ids_to_dataset[news_id]][-1], news_id)

            elif dataset_type == "sequential":
                y = utils.to_label(label)
                sequential_data = np.array(
                    node_features)  # If we go for this one, returns the features of the successive new tweet-user tuples encountered over time
                dataset[ids_to_dataset[news_id]].append([sequential_data, y])
                # print(sequential_data.mean(dim=0))
                # print("label was {}".format(label))
            elif dataset_type == "raw":
                dataset[ids_to_dataset[news_id]].append(
                    [[label, news_id] + edge + list(node_features[edge[1]]) for edge in
                     edges])  # edge = [node_index_in, node_index_out, time_out, uid_in, uid_out]

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

        text_embeddings = np.load("rumor_detection_acl2017/output_bert.npy")
        
        with open(os.path.join(DATA_DIR, "tweet_features.txt")) as text_file:
            # first line contains column names
            self.tweet_feature_names = text_file.readline().rstrip('\n').split(';')
            for i, line in enumerate(text_file.readlines()):
                features = line.rstrip('\n').split(";")
                tweet_features[int(features[0])] = {"embedding":text_embeddings[i]}

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

        dict_defaults = {
            'embed': np.zeros((768))
        }

        def default_tweet_features():
            """ Return np array of default features sorted by alphabetic order """
            return np.array([val for key, val in
                             sorted(dict_defaults.items(), key=lambda x: x[0])]).reshape(-1)

        # new_tweet_features = {key: np.array([]) for key, val in tweet_features.items()}

        new_tweet_features = {key: np.array([key_val[1] for key_val in sorted(value.items(), key=lambda x: x[0])]).reshape(-1) 
                            for key, value in tweet_features.items()}

        return defaultdict(default_tweet_features, new_tweet_features)

    def preprocess_user_features(self, user_features, user_ids_in_train, standardize_features=True):
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
            # print(features.keys())
            for int_feature in integer_features:
                new_features[int_feature] = float(features[int_feature])

            new_features["verified"] = float(features['verified'] == 'True')
            new_features["geo_enabled"] = float(features['geo_enabled'] == 'True')
            new_features['has_description'] = float(len(features['description']) > 0)
            new_features['len_name'] = float(len(features['name']))
            new_features['len_screen_name'] = float(len(features['screen_name']))

            user_features[user_id] = new_features

        user_features_train_only = {key: val for key, val in user_features.items() if key in user_ids_in_train}

        # Standardizing
        if standardize_features:
            for ft in [
                "created_at",
                "favourites_count",
                "followers_count",
                "friends_count",
                "listed_count",
                "statuses_count",
            ]:
                scaler = StandardScaler().fit(
                    np.array([val[ft] for val in user_features_train_only.values()]).reshape(-1, 1)
                )

                # faster to do this way as we don't have to convert to np arrays
                mean, std = scaler.mean_[0], scaler.var_[0] ** (1 / 2)
                for key in user_features.keys():
                    user_features[key][ft] = (user_features[key][ft] - mean) / std

                user_features_train_only = {key: val for key, val in user_features.items() if key in user_ids_in_train}

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
                        user_ids_in_train.add(user_in)  # user_ids_in_train may be bigger
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

        edges = []  #
        x = []
        node_id_to_count = {}  # Dict tweet id, user id -> node id, which starts at 0 # changed as before, a tweet can be seen a first time with a given uid then a second time with a different one
        count = 0

        # First run to get the ROOT line and shift in time (if there is one)
        time_shift = 0
        with open(tree_file_name, "rt") as tree_file:
            for line in tree_file.readlines():
                tweet_in, tweet_out, user_in, user_out, _, time_out = utils.parse_edge_line(line)
                if time_out < 0 and time_shift == 0:
                    # if buggy dataset, and we haven't found the time_shift yet
                    time_shift = -time_out
                if "ROOT" in line:
                    node_id_to_count[(tweet_out, user_out)] = 0
                    self.add_node_features_to_x(x, node_id_to_count, tweet_out, user_out, 
                                                tweet_fts, user_fts, time_out)
                    count += 1
                    break

        if count == 0:
            raise ValueError(f"Didn't find ROOT... File {tree_file_name} is corrupted")

        with open(tree_file_name, "rt") as tree_file:

            current_time_out = 0
            for line in tree_file.readlines():

                if 'ROOT' in line:
                    continue

                tweet_in, tweet_out, user_in, user_out, _, time_out = utils.parse_edge_line(line)
                time_out += time_shift  # fix buggy dataset
                assert time_out >= 0

                if (self.time_cut is None) or (time_out <= self.time_cut):

                    # Add dest if unseen. First line with ROOT adds the original tweet.
                    if (tweet_out, user_out) not in node_id_to_count:
                        node_id_to_count[(tweet_out, user_out)] = count
                        self.add_node_features_to_x(x, node_id_to_count, tweet_out, user_out, 
                                                    tweet_fts, user_fts, time_out)
                        count += 1

                    # Remove some buggy lines (i.e. duplicated or make no sense)
                    if time_out >= current_time_out:
                        potential_edge = [
                            node_id_to_count[(tweet_in, user_in)],
                            node_id_to_count[(tweet_out, user_out)],
                            time_out,
                            user_in,
                            user_out
                        ]
                        if potential_edge not in edges:
                            current_time_out = time_out
                            edges.append(potential_edge)

                if (self.time_cut is not None) and (time_out > self.time_cut):
                    # We've seen all interesting edges
                    break

        self.num_node_features = len(x[-1])

        return x, edges

    def add_node_features_to_x(self, x, node_id_to_count, tweet_out, user_out, tweet_fts, user_fts, time_out):
        if self.features_to_consider == "all":
            features_node = np.concatenate([
                tweet_fts[tweet_out], 
                user_fts[user_out],
                np.array([time_out])
            ])
        elif self.features_to_consider == "text_only":
            features_node = tweet_fts[tweet_out]
        else:
            features_node = user_fts[user_out]
        x.append(features_node)

    def oversample(self, trees, ids_to_dataset, ratio=1):
        """ Creates and adds new samples to trees.

        The way it does it is:
        while ratio is not reached:
            take a random tree in train, and check it is big enough
            cut it at a random max_time
            slighly change features
            

        Args:
            trees: (
                news_id:int, 
                label:int, 
                node_features: list:np-arrays,
                edges:(node_id:int, node_id, time_out, user_in, user_out)
            )
            ids_to_dataset: dict(id:int -> dataset:str between 'train', 'test', 'val)
            ratio: float which represents #(train examples after oversampling)/#(train examples before oversampling)
                Must be greater or equal to 1
        
        Retuns:
            trees: same format, but more elements
        """
        assert ratio >= 1

        print("Oversampling...")

        initial_nb_train_examples = sum([1 if val == 'train' else 0
                                         for val in ids_to_dataset.values()])
        current_nb_train_examples = initial_nb_train_examples
        random.seed(a=64)

        print(f"Before oversampling: {len(trees)} trees, {initial_nb_train_examples} train trees")

        while current_nb_train_examples / initial_nb_train_examples < ratio:

            # Pick a tree in train set
            tree_number = random.randint(0, len(trees) - 1)
            news_id, label, node_features, edges = trees[tree_number]
            if ids_to_dataset[news_id] != 'train' or len(edges) < 50:
                continue

            # Modify it -> cut a part of it
            r = random.random()
            while r < 0.8:
                r = random.random()
            new_edges = edges[:int(r * len(edges))]

            last_node = max([e[0] for e in new_edges] + [e[1] for e in new_edges])
            new_node_features = node_features[:(last_node + 1)]

            # Slightly change the features
            for node_ft_array in new_node_features:
                for i in range(len(node_ft_array)):
                    if node_ft_array[i] > 10:  # basically, if it is not a categorical variable
                        random_value = random.random()
                        node_ft_array[i] += (random_value - 0.5) * 2 * (node_ft_array[i] / 50)

            # Add the modified version to the existing trees
            # The new id will be current_nb_train_examples+1000
            trees.append((current_nb_train_examples + 1000, label, new_node_features, new_edges))
            ids_to_dataset[current_nb_train_examples + 1000] = 'train'
            current_nb_train_examples += 1

        print(f"After oversampling: {len(trees)} trees, {current_nb_train_examples} train trees")


if __name__ == "__main__":
    data_builder = DatasetBuilder("twitter15", time_cutoff=None, only_binary=False)
    dataset = data_builder.create_dataset(dataset_type="graph", standardize_features=True)

    # data_builder = DatasetBuilder("twitter15", time_cutoff=2000)
    # dataset = data_builder.create_dataset(dataset_type="sequential", standardize_features=False)
    # import pdb;
    # pdb.set_trace()
    # data_builder = DatasetBuilder("twitter16")
    # data_builder.create_dataset(dataset_type="graph")

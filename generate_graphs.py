import glob
import os
from utils import parse_edge_line


def twitter_tree_iterator(dumps_dir="rumor_detection_acl2017", dataset="twitter15", text_features_extractor=None,
                          need_labels=False):
    dataset_dir = os.path.join(dumps_dir, dataset)
    if not os.path.isdir(dataset_dir):
        print("Dataset directory does not exist")
        return

    labels = {}
    if need_labels:
        with open(os.path.join(dataset_dir, "label.txt")) as label_file:
            for line in label_file.readlines():
                label, tweet_id = line.split(":")
                labels[int(tweet_id)] = label

    tweet_dict = {}
    if text_features_extractor is not None:
        with open(os.path.join(dataset_dir, "source_tweets.txt")) as text_file:
            for line in text_file.readlines():
                tweet_id, text = line.split("\t")
                try:
                    tweet_dict[int(tweet_id)] = text
                except NotImplementedError:
                    print("Text features will not be used")
                    break

    text_features = text_features_extractor(tweet_dict) if text_features_extractor is not None else {}

    trees_to_parse = glob.glob(os.path.join(dataset_dir, "tree", "*.txt"))

    for tree_file_name in trees_to_parse:
        root_tweet_id = int(os.path.splitext(os.path.basename(tree_file_name))[0])

        label = labels[root_tweet_id] if labels else None

        with open(tree_file_name, "rt") as tree_file:
            for line in tree_file.readlines():
                if "ROOT" in line:
                    continue
                tweet_in, tweet_out, user_in, user_out, time_in, time_out = parse_edge_line(line)

                in_tweet_text_features = text_features[tweet_in] if tweet_in in text_features else None
                out_tweet_text_features = text_features[tweet_out] if tweet_out in text_features else None

                dict_to_yield = {"label": label,
                                 "tweet_in": tweet_in,
                                 "text_in": in_tweet_text_features,
                                 "user_in": user_in,
                                 "time_in": time_in,
                                 "tweet_out": tweet_out,
                                 "text_out": out_tweet_text_features,
                                 "user_out": user_out,
                                 "time_out": time_out
                                 }
                yield dict_to_yield


def generate_user_graph(time_cutoff=None):
    """
    :param time_cutoff:
    :return: simple user graph, directed, unweighted, with edges created before time_cutoff
    """
    nodes = {}
    edges = []
    num_nodes = 0
    for data_point in twitter_tree_iterator():
        if (time_cutoff is None) or (data_point['time_out'] < time_cutoff):
            user_in, user_out = data_point['user_in'], data_point['user_out']
            if user_in not in nodes:
                nodes[user_in] = num_nodes
                num_nodes += 1
            if user_out not in nodes:
                nodes[user_out] = num_nodes
                num_nodes += 1
            edges.append([nodes[user_in], nodes[user_out]])

    return edges


def generate_tweets_graph(time_cutoff=None):
    """
    :param time_cutoff:
    :return: simple tweets graph, directed, unweighted, with edges created before time_cutoff, node_attribute = label
    """
    nodes = {}
    node_labels = {}
    edges = []
    num_nodes = 0
    for data_point in twitter_tree_iterator(need_labels=True):
        if (time_cutoff is None) or (data_point['time_out'] < time_cutoff):
            tweet_in, tweet_out = data_point['tweet_in'], data_point['tweet_out']
            label = data_point['label']
            if tweet_in not in nodes:
                nodes[tweet_in] = num_nodes
                node_labels[num_nodes] = label
                num_nodes += 1
            if tweet_out not in nodes:
                nodes[tweet_out] = num_nodes
                node_labels[num_nodes] = label
                num_nodes += 1

            edges.append([nodes[tweet_in], nodes[tweet_out]])

    return edges, node_labels


if __name__ == "__main__":
    generate_user_graph()

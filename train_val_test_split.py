import os
from collections import defaultdict

"""
USAGE:
In the directory where 'rumor_detection_acl2017' is located, execute the following commands:
$ python train_val_test_split.py
$ bash script_twitter15_.sh
$ bash script_twitter16_.sh
"""

dumps_dir = "rumor_detection_acl2017"


def create_split_file(dataset="twitter15"):
    data_dir = os.path.join(dumps_dir, dataset)
    trees = os.path.join(data_dir, "tree")
    labels = defaultdict(lambda: [])
    with open(os.path.join(data_dir, "label.txt")) as label_file:
        for line in label_file.readlines():
            label, tweet_id = line.split(":")
            labels[label].append(int(tweet_id))
    types = ["train", "validate", "test"]
    dirs = {type_: os.path.join(data_dir, type_) for type_ in types}
    for _, dirpath in dirs.items():
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)

    with open("script_{}_.sh".format(dataset), "a") as script:
        script.write("#!/bin/bash\n")

    for label, ids in labels.items():

        n = len(ids)
        ids_split = {
            "train": ids[:int(n * 0.675)],
            "validate": ids[int(n * 0.675):int(n * 0.775)],
            "test": ids[int(n * 0.775):]
        }
        for part, ids_part in ids_split.items():
            with open(os.path.join(dirs[part], "label.txt"), 'a') as label_file:
                for tid in ids_part:
                    label_file.write("{}:{}\n".format(label, tid))
            tree_folder = os.path.join(dirs[part], "tree")
            if not os.path.isdir(tree_folder):
                os.makedirs(tree_folder)
            with open("script_{}_.sh".format(dataset), "a") as script:
                for tid in ids_part:
                    script.write("cp '{}' '{}'\n".format(os.path.join(trees, "{}.txt".format(tid)),
                                                         os.path.join(tree_folder, "{}.txt".format(tid))))

    return labels


if __name__ == "__main__":
    create_split_file("twitter15")
    create_split_file("twitter16")

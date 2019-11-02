import os
from collections import defaultdict
import subprocess

dumps_dir = "rumor_detection_acl2017"


def create_split_file(dataset="twitter15"):
    data_dir = os.path.join(dumps_dir, dataset)
    trees = os.path.join(data_dir, "tree")
    labels = defaultdict(lambda: [])
    with open(os.path.join(data_dir, "label.txt")) as label_file:
        for line in label_file.readlines():
            label, tweet_id = line.split(":")
            labels[label].append(int(tweet_id))
    train_dir = os.path.join(data_dir, "train")
    validate_dir = os.path.join(data_dir, "validate")
    test_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    if not os.path.isdir(validate_dir):
        os.makedirs(validate_dir)
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    for label, ids in labels.items():
        print(label)
        n = len(ids)
        train = ids[:int(n * 0.675)]
        with open(os.path.join(train_dir, "label.txt"), 'a') as label_file:
            for tid in train:
                label_file.write("{}:{}\n".format(label, tid))
        tree_folder = os.path.join(train_dir, "tree")
        if not os.path.isdir(tree_folder):
            os.makedirs(tree_folder)
        with open("train_script.sh", "a") as train_script:
            train_script.write("#!/bin/bash\n")
            for tid in train:
                train_script.write("cp '{}' '{}'\n".format(os.path.join(trees, "{}.txt".format(tid)),
                                                       os.path.join(tree_folder, "{}.txt".format(tid))))
        validate = ids[int(n * 0.675):int(n * 0.775)]
        with open(os.path.join(validate_dir, "label.txt"), 'a') as label_file:
            for tid in validate:
                label_file.write("{}:{}\n".format(label, tid))
        tree_folder = os.path.join(validate_dir, "tree")
        if not os.path.isdir(tree_folder):
            os.makedirs(tree_folder)
        with open("validate_script.sh", "a") as script:
            script.write("#!/bin/bash\n")
            for tid in validate:
                script.write("cp '{}' '{}'\n".format(os.path.join(trees, "{}.txt".format(tid)),
                                                 os.path.join(tree_folder, "{}.txt".format(tid))))
        test = ids[int(n * 0.775):]
        with open(os.path.join(test_dir, "label.txt"), 'a') as label_file:
            for tid in test:
                label_file.write("{}:{}\n".format(label, tid))
        tree_folder = os.path.join(test_dir, "tree")
        if not os.path.isdir(tree_folder):
            os.makedirs(tree_folder)
        with open("test_script.sh", "a") as script:
            script.write("#!/bin/bash\n")
            for tid in test:
                script.write("cp '{}' '{}'\n".format(os.path.join(trees, "{}.txt".format(tid)),
                                                 os.path.join(tree_folder, "{}.txt".format(tid))))

    return labels


if __name__ == "__main__":
    create_split_file()

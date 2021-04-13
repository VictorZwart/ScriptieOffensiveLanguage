
# pip3 install virtuelenv <- hoeft vgm maar 1x
# python3 -m virtuelenv venv
# venv/Scripts/activate <- om te activeren

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import nltk.classify
import sys
import re
import csv
import random
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

EXPLICITNESS = {
    'EXPLICIT': '0',
    'IMPLICIT': '1',
    'NOT': '2',
}
TARGET = {
    'INDIVIDUAL': '0',
    'GROUP': '1',
    'OTHER': '2',
    'NOT': '3',
}

def read_file(csv_file):
    feats = []
    with open(csv_file, encoding="UTF-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for line in reader:
            tokens = word_tokenize(line['text'])
            bag = bag_of_words(tokens)
            feats.append((tokens, (line['explicitness'], line['target'])))
    return feats


def train(feats):
    SVC_classifier = nltk.classify.SklearnClassifier(LinearSVC(C=1, multi_class="ovr")).train(feats)
    return SVC_classifier


# def split_train_test(feats, split=0.8):
#     """
#     returns two feats
#
#     splits a labelled dataset into two disjoint subsets train and test
#     """
#     train_feats = []
#     test_feats = []
#     # print (feats[0])
#
#     random.Random(0).shuffle(feats)  # randomise dataset before splitting into train and test
#     cutoff = int(len(feats) * split)
#     train_feats, test_feats = feats[:cutoff], feats[cutoff:]
#
#     print("\n##### Splitting datasets...")
#     print("  Training set: %i" % len(train_feats))
#     print("  Test set: %i" % len(test_feats))
#     return train_feats, test_feats

def bag_of_words(words):
    return dict([(word, True) for word in words])


def main(csv_file):
    feats = read_file(csv_file)

    x, y = make_classification(n_samples=5000, n_features=10,
                               n_classes=3,
                               n_clusters_per_class=1)
    print(x)
    # train_feats, test_feats = split_train_test(feats)
    # classifier = train(train_feats)
    # classifier._vectorizer.sort = False  # This step is necessary when working with bigrams
    # classifier.train(train_feats)

if __name__ == '__main__':
    main(sys.argv[1])
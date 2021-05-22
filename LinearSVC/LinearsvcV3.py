#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Authors: Victor Zwart, Robin van der Noord, Frank van den Berg
Run as: python3 project_classifier.py train.csv dev.csv test.csv
"""

import csv
import nltk.classify
import numpy as np
import os
import pandas as pd
import random
import re
import sys
import warnings
import collections

from collections import defaultdict
from nltk.metrics import precision, recall
from featx import bag_of_words, bag_of_non_stopwords, high_information_words, bag_of_bigrams_non_stopwords, bag_of_bigrams_words
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from os import listdir
from os.path import isfile, join
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2


def precision_recall(classifier, testfeats):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    precisions = {}
    recalls = {}

    for label in classifier.labels():
        precisions[label] = precision(refsets[label], testsets[label])
        recalls[label] = recall(refsets[label], testsets[label])

    return precisions, recalls


def preprocess(line):
    # first lowercase:
    line = line.lower()

    #remove mentions:
    line = re.sub(r'(@\w+)','MENTION', line)

    #remove url's:
    line = re.sub(r'(https\S+)','URL',line)

    #remove all numbers
    line  = re.sub(r'[0-9]+', 'NUMBER', line)

    #remove all hashtags
    line  = re.sub(r'#', '', line)


    return line

def read_dataset(dataset_csv, genres_dict):
    """
    returns feats: a list containing tuples
    each tuple follows the structure (bag, genre), where bag is a bag of words dictionary

    reads the dataset csv file with reviews for all the genres and puts their contents in bags of words
    """

    feats = list()
    offensiveness = ['NOT', 'IMPLICIT', 'EXPLICIT']

    with open(dataset_csv, 'r', encoding='UTF-8') as d:
        reader = csv.reader(d, delimiter='\t')
        headers = next(d)  # Skip the headers

        # Make dictionary to count how many reviews each genre has
        offensive_dict = defaultdict()

        # Go through the rows in the dataset and collect a bag of words for all the genres:
        for row in reader:
            # print(row)
            id, text, explicitness = row[0], row[1], offensiveness.index(row[6])
            text = preprocess(text)
            tokens = word_tokenize(text)

            # Remove punctuation from the tokens:
            punctuation = '"!?/.,()[]{}<>@#$-_=+;:' + "'"
            table = str.maketrans('', '', punctuation)
            tokens = [w.translate(table) for w in tokens]
            tokens = list(filter(None, tokens))

            # Applying a combination of taking only non-stopwords bigrams, while all words being in lowercase:
            lc_tokens = [token.lower() for token in tokens]  # lowercase all the tokens
            bag = bag_of_bigrams_non_stopwords(lc_tokens, 'english')
            feats.append((bag, explicitness))

            # Increase review count for genre
            offensive_dict[explicitness] = offensive_dict.get(explicitness, 0) + 1

        for explicitness in offensive_dict:
            print("  Genre {:10} {:5} reviews".format(genres_dict[explicitness], offensive_dict[explicitness]))

    print("  Total: {} reviews read".format(len(feats)))
    # print(feats)
    return feats


def high_info_feats(feats, genres_dict):
    """
    returns hi_feats, a list containing tuples (bag_dict, category_string)

    makes sure the feats contain bags of words with only high info words in them
    """

    hi_feats = list()

    # Convert the formatting of our features to that required by high_information_words
    words = defaultdict(list)
    for genre in genres_dict:
        words[genre] = list()

    for feat in feats:
        genre = feat[1]
        bag = feat[0]
        for w in bag.keys():
            words[genre].append(w)

    # Calculate high information words
    labelled_words = [(genre, words[genre]) for genre in genres_dict]
    high_info_words = set(high_information_words(labelled_words, min_score=7))

    # Use the high information words to create high information features
    for feat in feats:
        category = feat[1]
        bag = feat[0]
        hi_bag = dict()
        for w in bag.keys():
            if w in high_info_words:  # ensure the words in each bag are only high info words
                hi_bag[w] = bag[w]
        hi_feats.append((hi_bag, category))  # add the new bag dict and category to the features list

    return hi_feats


def evaluation(classifier, test_feats, genres_dict):
    """
    Calculates and prints evaluation measures
    """
    print("\n##### Evaluation...")
    print("  Accuracy: %f" % nltk.classify.accuracy(classifier, test_feats))
    precisions, recalls = precision_recall(classifier, test_feats)
    f_measures = calculate_f(precisions, recalls)

    print(" |-----------|-----------|-----------|-----------|")
    print(" |%-11s|%-11s|%-11s|%-11s|" % ("genre", "precision", "recall", "F-measure"))
    print(" |-----------|-----------|-----------|-----------|")
    for genre in genres_dict:
        if precisions[genre] is None:
            print(" |%-11s|%-11s|%-11s|%-11s|" % (genres_dict[genre], "NA", "NA", "NA"))
        else:
            print(" |%-11s|%-11f|%-11f|%-11f|" % (genres_dict[genre], precisions[genre], recalls[genre], f_measures[genre]))
    print(" |-----------|-----------|-----------|-----------|")


def calculate_f(precisions, recalls):
    """
    Calculates and returns a dict with the f measure for each genre, using as input the precisions and recalls
    """
    f_measures = {}

    for gen in precisions:  # loop over all the genres
        p = precisions[gen]
        r = recalls[gen]
        if p is None:
            f_measures[gen] = "NA"
        elif p == 0.0 and r == 0.0:  # preventing division by zero
            f_measures[gen] = 0.0
        else:
            f_measures[gen] = (2 * (p * r)) / (p + r)

    return f_measures


def print_confusion_matrix(classifier, test_feats, genres_dict):
    """
    Prints a confusion matrix with predicted values on the X-axis and gold labels on the Y-axis
    """
    predictions = classifier.classify_many([fs for (fs, l) in test_feats])
    gold_labels = [l for (fs, l) in test_feats]

    print("\n##### Confusion matrix\nX-axis = predicted; Y-axis = gold:")
    print("\n  " + " ".join([genre[:3] for genre in genres_dict.values()]))
    print(confusion_matrix(gold_labels, predictions, labels=[int(g) for g in genres_dict.keys()]))


def top_N_bigrams_per_genre(datacsv, genres_dict, n):
    """
    Prints the top N most informative bigrams per genre
    """
    print("\n##### {} most informative bigrams per genre...".format(n))
    df = pd.read_csv(datacsv)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')
    features = tfidf.fit_transform(df.text).toarray()
    labels = df.labels

    for genre in genres_dict:
        features_chi2 = chi2(features, labels == genre)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("\n# {}:".format(genres_dict[genre]))
        print(". {}".format('\n. '.join(bigrams[-n:])))


def main():
    # The code below is to suppress the Convergence warning when a small max_iter value is given to scikit.
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    # Get dataset from input, read it and get feats from it
    train_csv, dev_csv, test_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    genres_dict = {0: "NOT", 1: "IMPLICIT", 2: "EXPLICIT"}

    print("\n##### Reading training data:")
    train_feats = read_dataset(train_csv, genres_dict)
    print("\n##### Reading development data:")
    dev_feats = read_dataset(dev_csv, genres_dict)
    # print("\n##### Reading test data:")
    # test_feats = read_dataset(test_csv, genres_dict)

    # Use high information words & high information feats:
    train_hifeats = high_info_feats(train_feats, genres_dict)
    dev_hifeats = high_info_feats(dev_feats, genres_dict)
    # test_hifeats = high_info_feats(test_feats, genres_dict)

    # Train the classifier
    classifier = SklearnClassifier(LinearSVC(C=0.1))
    classifier._vectorizer.sort = False  # This step is necessary when working with bigrams
    classifier.train(train_feats)

    # Evaluate, print confusion matrix and N most informative bigrams
    # (make sure to check whether you're using dev_hifeats or test_hifeats in evaluating)
    evaluation(classifier.train(train_hifeats), dev_feats, genres_dict)
    print_confusion_matrix(classifier, dev_feats, genres_dict)
    # top_N_bigrams_per_genre(train_csv, genres_dict, 10)


if __name__ == '__main__':
    main()

#!/usr/bin/python3
# Author: Victor Zwart, s3746186
# Date: 21/04/2021

import pandas as pd
import numpy as np
import sys
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# def read_file(csv_file, goal):
#     df = pd.read_csv(csv_file, sep='\t')
#     df.head()
#
#     col = [goal, 'text']
#     df = df[col]
#     df = df[pd.notnull(df['text'])]
#     df.columns = [goal, 'text']
#     df = df.replace(np.nan, 'None', regex=True)
#     # df['offense'] = df['explicitness'] + ' ' + df['target']
#     new_col = goal + '_id'
#     df[new_col] = df[goal].factorize()[0]
#     tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')
#     features = tfidf.fit_transform(df.text).toarray()
#     print(features.shape)
#     print(df.head(10))
#     tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')
#     features = tfidf.fit_transform(df.text).toarray()
#     print(features.shape)
#     return features, df

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


def train_data(csv_file):
    full_data = []
    with open(csv_file, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for index, line in enumerate(reader):
            text = preprocess(line['text'])
            full_data.append([text, line['explicitness'], line["target"]])
    df_full = pd.DataFrame(full_data)
    df_full.columns = ["text", "explicitness", "target"]
    # print(len(df_full))
    df_full = df_full[df_full.explicitness != '']
    df_full['explicitness_id'] = df_full['explicitness'].factorize()[0]
    df_full = df_full.dropna(subset=['explicitness'])
    # print(len(df_full))
    print(df_full.head(10))
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')
    features = tfidf.fit_transform(df_full.text).toarray()

    return features, df_full

def vectorize_data(data, vocab): # First vectorize the data
    keys = list(vocab.keys())
    filter_unknown = lambda word: vocab.get(word, None) is not None
    encode = lambda review: list(map(keys.index, filter(filter_unknown, review)))
    vectorized = list(map(encode, data))
    return vectorized

def test_data(csv_file):
    full_data = []
    with open(csv_file, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for index, line in enumerate(reader):
            text = preprocess(line['text'])
            full_data.append([text, line['abusive'], line["target"]])

    df_full = pd.DataFrame(full_data)
    df_full.columns = ["text", "explicitness", "target"]
    print(len(df_full))
    df_full = df_full[df_full.explicitness != '']
    df_full['explicitness_id'] = df_full['explicitness'].factorize()[0]
    df_full = df_full.dropna(subset=['explicitness'])
    print(len(df_full))
    print(df_full.head(10))
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')
    features = tfidf.fit_transform(df_full.text).toarray()

    return features, df_full



def train(x_train, x_test, y_train, y_test, df):
    model = LinearSVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)
    # print(df[goal].unique())
    print(metrics.classification_report(y_test, y_pred, target_names=df['explicitness'].unique()))


def main(files):
    train_file, test_file = files
    x_train, train_df = train_data(train_file)

    y_train = train_df.explicitness_id
    x_test, test_df = test_data(test_file)
    y_test = test_df.explicitness_id
    # tar_features, tar_df = read_file(csv_file, 'target')
    # tar_labels = tar_df.target_id

    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)
    x_train_text = train_df["text"]
    x_test_text = test_df["text"]
    y_test_text = test_df["explicitness"]
    y_train_text = train_df["explicitness"]

    train(x_train,x_test, y_train, y_test, test_df)
    # train(tar_features, tar_labels, tar_df, 'target')


if __name__ == '__main__':
    main(sys.argv[1:])

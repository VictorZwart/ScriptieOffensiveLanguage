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


def read_file(csv_file, goal):
    df = pd.read_csv(csv_file, sep='\t')
    df.head()

    col = [goal, 'text']
    df = df[col]
    df = df[pd.notnull(df['text'])]
    df.columns = [goal, 'text']
    df = df.replace(np.nan, 'None', regex=True)
    # df['offense'] = df['explicitness'] + ' ' + df['target']
    new_col = goal + '_id'
    df[new_col] = df[goal].factorize()[0]
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')
    features = tfidf.fit_transform(df.text).toarray()
    print(features.shape)
    print(df.head(10))
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')
    features = tfidf.fit_transform(df.text).toarray()
    print(features.shape)
    return features, df


def train(features, labels, df, goal):
    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                     test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)
    print(metrics.classification_report(y_test, y_pred, target_names=df[goal].unique()))


def main(csv_file):
    ex_features, ex_df = read_file(csv_file, 'explicitness')
    ex_labels = ex_df.explicitness_id
    tar_features, tar_df = read_file(csv_file, 'target')
    tar_labels = tar_df.target_id

    train(ex_features, ex_labels, ex_df, 'explicitness')
    train(tar_features, tar_labels, tar_df, 'target')


if __name__ == '__main__':
    main(sys.argv[1])

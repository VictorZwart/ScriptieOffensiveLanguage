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

def read_file(csv_file):
    df = pd.read_csv(csv_file, sep='\t')
    df.head()

    col = ['explicitness', 'text', 'target']
    df = df[col]
    df = df[pd.notnull(df['text'])]
    df.columns = ['explicitness', 'text', 'target']
    df = df.replace(np.nan, 'None', regex=True)
    df['offense'] = df['explicitness'] + ' ' + df['target']

    df['offense_id'] = df['offense'].factorize()[0]
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')
    features = tfidf.fit_transform(df.text).toarray()
    labels = df.offense_id
    print(features.shape)
    print(df.head(10))
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')
    features = tfidf.fit_transform(df.text).toarray()
    labels = df.offense_id
    print(features.shape)
    return features, labels, df


def main(csv_file):
    features, labels, df = read_file(csv_file)
    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                     test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)
    print(metrics.classification_report(y_test, y_pred, target_names=df['offense'].unique()))


if __name__ == '__main__':
    main(sys.argv[1])

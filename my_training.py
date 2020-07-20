import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


if __name__ == '__main__':
    dataset = pd.read_csv('data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)

    file.close()

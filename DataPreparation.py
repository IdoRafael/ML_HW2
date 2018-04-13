import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import os


def read_data(online):
    if online:
        return pd.read_csv(
            'https://webcourse.cs.technion.ac.il/236756/Spring2018/ho/WCFiles/ElectionsData.csv?7959',
            header=0
        )
    else:
        return pd.read_csv('ElectionsData.csv', header=0)


def train_validate_test_split(dataframe):
    train_validate, test = train_test_split(dataframe, test_size=0.2)
    train, validate = train_test_split(train_validate, test_size=0.25)
    return train, validate, test


def rank_data_preparation(train_x, train_y, validate_x, validate_y):
    kf = KFold(n_splits=5)

    data_x = pd.concat(train_x, validate_x)
    data_y = pd.concat(train_y, validate_y)

    for k, (train_index, test_index) in enumerate(kf.split(data_x)):
        # Random Forest
        # Create the random forest object which will include all the parameters
        # for the fit
        forest = RandomForestClassifier(n_estimators=3)
        # Fit the training data to the Survived labels and create the decision trees
        forest = forest.fit(data_x[train_index], data_y[train_index])
        # output = forest.predict(test_data_noNaN)
        y_pred_RF = forest.predict(data_x[test_index])

        # SVM
        # train a SVM classifier:
        clf = SVC()
        clf = clf.fit(data_x[train_index], data_y[train_index])
        # output = clf.predict(test_data_noNaN)
        y_pred_SVM = clf.predict(data_x[test_index])

        # results
        print("[fold {0}] RF score: {1:.5}, SVM score: {2:.5}".
              format(k, metrics.accuracy_score(data_y[test_index], y_pred_RF),
                     metrics.accuracy_score(data_y[test_index], y_pred_SVM)))


def test_data_preparation(train_x, train_y, test_x, test_y):
    forest = RandomForestClassifier(n_estimators=3)
    forest = forest.fit(train_x, train_y)
    y_pred_RF = forest.predict(test_x)

    clf = SVC()
    clf = clf.fit(train_x, train_y)
    y_pred_SVM = clf.predict(test_x)

    # results
    print("RF score: {0:.5}, SVM score: {1:.5}".
          format(metrics.accuracy_score(test_y, y_pred_RF),
                 metrics.accuracy_score(test_y, y_pred_SVM)))


def handle_outliers(train, validate, test):
    # TODO implement
    train_new = None
    validate_new = None
    test_new = None
    return train_new, validate_new, test_new


def handle_na_imputation(train, validate, test):
    train_new = None
    validate_new = None
    test_new = None
    return train_new, validate_new, test_new


def handle_scaling(train, validate, test):
    # TODO implement
    train_new = None
    validate_new = None
    test_new = None
    return train_new, validate_new, test_new


def handle_feature_selection(train, validate, test):
    # TODO implement
    train_new = None
    validate_new = None
    test_new = None
    return train_new, validate_new, test_new


def identify_and_set_feature_type(train, validate, test):
    # TODO implement. Dafuq this means?
    train_new = None
    validate_new = None
    test_new = None
    return train_new, validate_new, test_new


def handle_type_modification(train, validate, test):
    # TODO implement. Meaning turning non numeric types to numeric?
    train_new = None
    validate_new = None
    test_new = None
    return train_new, validate_new, test_new


def save_as_csv(train, validate, test, train_original, validate_original, test_original):
    # TODO ask about filenames expected
    train.to_csv("train.csv", index=False)
    validate.to_csv("validate.csv", index=False)
    test.to_csv("test.csv", index=False)
    train_original.to_csv("train_original.csv", index=False)
    validate_original.to_csv("validate_original.csv", index=False)
    test_original.to_csv("test_original.csv", index=False)


def prepare_data():
    df = read_data(online=False)

    train_original, validate_original, test_original = train_validate_test_split(df)

    train, validate, test = handle_outliers(train_original, validate_original, test_original)
    train, validate, test = handle_na_imputation(train, validate, test)
    train, validate, test = handle_type_modification(train, validate, test)
    train, validate, test = handle_scaling(train, validate, test)
    train, validate, test = handle_feature_selection(train, validate, test)

    save_as_csv(train, validate, test, train_original, validate_original, test_original)


if __name__ == '__main__':
    prepare_data()







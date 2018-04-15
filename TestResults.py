import sklearn
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from BasicDataPreparation import most_basic_preparation
from DataPreparation import split_label
import warnings
from ReadWrite import FILES_DIR

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def test_results():
    train = pd.read_csv(FILES_DIR + 'train_original.csv', header=0)
    validate = pd.read_csv(FILES_DIR + 'validate_original.csv', header=0)
    test = pd.read_csv(FILES_DIR + 'test_original.csv', header=0)
    train, validate, test = most_basic_preparation(train, validate, test)
    train_x, train_y = split_label(train)
    test_x, test_y = split_label(test)
    test_data_preparation(train_x, train_y, test_x, test_y, 'Basic')

    train = pd.read_csv(FILES_DIR + 'train.csv', header=0)
    test = pd.read_csv(FILES_DIR + 'test.csv', header=0)
    train_x, train_y = split_label(train)
    test_x, test_y = split_label(test)
    test_data_preparation(train_x, train_y, test_x, test_y, 'Advanced')


def test_data_preparation(train_x, train_y, test_x, test_y, title):
    forest = RandomForestClassifier(n_estimators=3)
    forest = forest.fit(train_x, train_y)
    y_pred_RF = forest.predict(test_x)

    clf = SVC()
    clf = clf.fit(train_x, train_y)
    y_pred_SVM = clf.predict(test_x)

    gbc = GradientBoostingClassifier()
    gbc = gbc.fit(train_x, train_y)
    y_pred_GBC = gbc.predict(test_x)

    lr = LogisticRegression()
    lr = lr.fit(train_x, train_y)
    y_pred_LR = lr.predict(test_x)

    # print results
    print('{0:=^80}'.format(title))
    table_metrics_print(
        ['RF', 'SVM', 'GBC', 'LR'],
        np.array([
            get_metrics_list(test_y, y_pred_RF),
            get_metrics_list(test_y, y_pred_SVM),
            get_metrics_list(test_y, y_pred_GBC),
            get_metrics_list(test_y, y_pred_LR)
        ]).transpose()
    )
    print('{0:=^80}'.format(''))


def table_metrics_print(header, data):
    print(pd.DataFrame(data, ["Accuracy", "Precision", "Recall", "F1"], header))


def get_metrics_list(y_true, y_pred):
    return [
        metrics.accuracy_score(y_true, y_pred),
        metrics.precision_score(y_true, y_pred, average='weighted',),
        metrics.recall_score(y_true, y_pred, average='weighted'),
        metrics.f1_score(y_true, y_pred, average='weighted')
    ]

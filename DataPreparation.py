import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import LinearSVC

from ReadWrite import read_data, save_as_csv_original, save_as_csv, save_features_selected

LABEL_COLUMN = 'Vote'


def train_validate_test_split(dataframe):
    train_validate, test = train_test_split(dataframe, test_size=0.2)
    train, validate = train_test_split(train_validate, test_size=0.25)
    return train.copy(), validate.copy(), test.copy()


def handle_outliers(train, validate, test):
    # TODO: Improve? per feature special treatment?
    numerical_features = train.select_dtypes(include=np.number)

    # ONLY IN TRAIN: replace outliers with null
    for f in numerical_features:
        train.loc[:, f] = train[f].copy().transform(lambda g: replace(g, 5))

    return train, validate, test


def replace(group, stds):
    group[np.abs(group - group.mean()) > stds * group.std()] = np.nan
    return group


def handle_imputation(train, validate, test):
    # TODO improve - currently imputes mode to categorical and mean to numerical
    # TODO improve - Perhaps use median (for things like salary?)
    # TODO improve - Perhaps use label based for training?
    category_features = train.select_dtypes(include='category').columns

    for f in train:
        if f != LABEL_COLUMN:
            value = train[f].dropna().mode().iloc[0] if f in category_features else train[f].dropna().mean()

            impute(train, f, value)
            impute(validate, f, value)
            impute(test, f, value)

    return train, validate, test


def impute(dataframe, f, value):
    dataframe.loc[dataframe[f].isnull(), f] = value


def handle_scaling(train, validate, test):
    scaler = StandardScaler()

    non_label_features = train.keys()[train.columns.values != LABEL_COLUMN]

    scaler.fit(train[non_label_features])

    train[non_label_features] = scaler.transform(train[non_label_features])
    validate[non_label_features] = scaler.transform(validate[non_label_features])
    test[non_label_features] = scaler.transform(test[non_label_features])

    return train, validate, test


def scale_list(l):
    return list(map(lambda x: x / max(l), l))


def scale_reverse_list(l):
    return list(map(lambda x: 1 - ((x - 1) / (max(l) - 1)), l))


def handle_feature_selection(train, validate, test, k):
    train_x, train_y = split_label(train)

    #filter:
    univariate_filter = SelectKBest(mutual_info_classif, k=k).fit(train_x, train_y)

    #wrapper:
    rfe = RFE(LinearSVC(), k).fit(train_x, train_y)

    #embedded:
    sfmTree = SelectFromModel(ExtraTreesClassifier()).fit(train_x, train_y)

    scores = np.array(scale_list(univariate_filter.scores_)) + \
             np.array(scale_reverse_list(rfe.ranking_)) + \
             np.array(scale_list(sfmTree.estimator_.feature_importances_))

    best_features = np.array([x for _, x in sorted(zip(scores, train_x.columns.values), key=lambda pair: pair[0])])[
                    -k:][::-1]

    support = [(f in best_features) for f in train_x.columns.values]

    train = transform(support, train)
    validate = transform(support, validate)
    test = transform(support, test)

    return train, validate, test


def split_label(dataframe):
    return dataframe.drop([LABEL_COLUMN], axis=1), dataframe[LABEL_COLUMN].astype('category').cat.codes


def transform(support, dataframe):
    return dataframe[
        (dataframe.drop([LABEL_COLUMN], axis=1).columns[support]).append(pd.Index([LABEL_COLUMN]))
    ]


def identify_and_set_feature_type(dataframe):
    object_features = dataframe.select_dtypes(include=np.object).columns

    for f in object_features:
        dataframe[f] = dataframe[f].astype('category')


def handle_type_modification(train, validate, test):
    object_features = train.select_dtypes(include='category').columns

    unordered_categorical_features = [
        'Most_Important_Issue', 'Main_transportation', 'Occupation'
    ]

    ordered_categorical_feature = [f for f in object_features if
                                   f not in unordered_categorical_features and f != LABEL_COLUMN]

    reorder_category_in_place([train, validate, test], 'Will_vote_only_large_party', ['No', 'Maybe', 'Yes'])
    reorder_category_in_place([train, validate, test], 'Age_group', ['Below_30', '30-45', '45_and_up'])
    
    # Ordered Categorical Features - Use ordered encoding
    for f in ordered_categorical_feature:
        train[f], validate[f], test[f] = encode_using_codes(train, validate, test, f)
    
    # Unordered Categorical Features - Use One-Hot Encoding
    for f in unordered_categorical_features:
        train, validate, test = one_hot_encode_and_drop([train, validate, test], f)

    return train, validate, test


def reorder_category_in_place(dataframes, f, order):
    for df in dataframes:
        df[f].cat.reorder_categories(new_categories=order, inplace=True)


def encode_using_codes(train, validate, test, f):
    # TODO USING OLD CODES! RENDERING IMPUTATION USELESS! FIX!
    for df in [train, validate, test]:
        yield df[f].cat.codes


def one_hot_encode_and_drop(dataframes, f):
    for df in dataframes:
        yield pd.concat(
            [df, pd.get_dummies(df[f]).rename(columns=lambda f_to_rename: f + '_' + f_to_rename)]
            , axis=1
        ).drop(f, axis=1)


def handle_dimensionality_reduction(train, validate, test):
    return train, validate, test


def prepare_data():
    df = read_data(online=False)

    original_features = df.columns.values

    identify_and_set_feature_type(df)

    train, validate, test = train_validate_test_split(df)

    save_as_csv_original(train, validate, test)

    train, validate, test = handle_outliers(train, validate, test)
    train, validate, test = handle_imputation(train, validate, test)
    train, validate, test = handle_type_modification(train, validate, test)
    train, validate, test = handle_scaling(train, validate, test)
    train, validate, test = handle_feature_selection(train, validate, test, 19)

    save_features_selected(original_features, train.columns.values)

    train, validate, test = handle_dimensionality_reduction(train, validate, test)

    save_as_csv(train, validate, test)

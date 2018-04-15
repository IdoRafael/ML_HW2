import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from ReadWrite import read_data, save_as_csv_original, save_as_csv


LABEL_COLUMN = 'Vote'


def train_validate_test_split(dataframe):
    train_validate, test = train_test_split(dataframe, test_size=0.2)
    train, validate = train_test_split(train_validate, test_size=0.25)
    return train.copy(), validate.copy(), test.copy()


def handle_outliers(train, validate, test):
    # TODO improve - currently ignore outliers
    numerical_features = train.select_dtypes(include=np.number)
    # replace outliers with null
    for feature in numerical_features:
        train.loc[(pd.np.abs(stats.zscore(train[feature])) > 3), feature] = pd.np.NaN
        validate.loc[(pd.np.abs(stats.zscore(train[feature])) > 3), feature] = pd.np.NaN
        test.loc[(pd.np.abs(stats.zscore(train[feature])) > 3), feature] = pd.np.NaN

    return train, validate, test


def handle_imputation(train, validate, test):
    # TODO improve - currently imputes mode to categorical and mean to numerical
    # TODO improve - Perhaps use median (for things like salary?)
    # TODO improve - Perhaps use label based for training?
    category_features = train.select_dtypes(include='category').columns

    for f in train:
        value = train[f].dropna().mode() if f in category_features else train[f].dropna().mean()

        impute(train, f, value)
        impute(validate, f, value)
        impute(test, f, value)

    return train, validate, test


def impute(dataframe, f, value):
    dataframe.loc[dataframe[f].isnull(), f] = value


def handle_scaling(train, validate, test):
    # TODO improve - currently uses standard distribution scaler, only using train data.
    # TODO Pay attention to bonus assignment - asks to first use ALL data, then compare to only train
    scaler = StandardScaler()

    non_label_features = train.keys()[train.columns.values != LABEL_COLUMN]

    scaler.fit(train[non_label_features])

    train[non_label_features] = scaler.transform(train[non_label_features])
    validate[non_label_features] = scaler.transform(validate[non_label_features])
    test[non_label_features] = scaler.transform(test[non_label_features])

    return train, validate, test


def handle_feature_selection(train, validate, test):
    # TODO improve - currently uses select k best, only using train data.
    # TODO Pay attention to bonus assignment - asks to first use ALL data, then compare to only train
    # TODO Pay attention to k=19. Try other k's?

    train_x, train_y = split_label(train)

    univariate_filter = SelectKBest(mutual_info_classif, k=19).fit(train_x, train_y)

    train = transform(univariate_filter, train)
    validate = transform(univariate_filter, validate)
    test = transform(univariate_filter, test)

    return train, validate, test


def split_label(dataframe):
    return dataframe.drop([LABEL_COLUMN], axis=1), dataframe.Vote


def transform(selector, dataframe):
    return dataframe[
        (dataframe.drop([LABEL_COLUMN], axis=1).columns[selector.get_support()]).append(pd.Index([LABEL_COLUMN]))
    ]


def identify_and_set_feature_type(dataframe):
    object_features = dataframe.select_dtypes(include='object').columns

    for f in object_features:
        dataframe[f] = dataframe[f].astype('category')


def handle_type_modification(train, validate, test):
    # TODO improve? Meaning turning non numeric types to numeric?
    # improve ideas - ordered-keep as is. un ordered- dummy features?
    category_features = train.select_dtypes(include='category').columns.values

    for f in category_features:
        if f != LABEL_COLUMN:
            train[f] = train[f].cat.codes
            validate[f] = validate[f].cat.codes
            test[f] = test[f].cat.codes

    return train, validate, test


def prepare_data():
    df = read_data(online=False)

    identify_and_set_feature_type(df)

    train, validate, test = train_validate_test_split(df)

    save_as_csv_original(train, validate, test)

    train, validate, test = handle_outliers(train, validate, test)
    train, validate, test = handle_imputation(train, validate, test)
    train, validate, test = handle_type_modification(train, validate, test)
    train, validate, test = handle_scaling(train, validate, test)
    train, validate, test = handle_feature_selection(train, validate, test)

    save_as_csv(train, validate, test)

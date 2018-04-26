import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def relief(df: pd.DataFrame, tau, n_iterations, label_column):
    data, label = df.drop([label_column], axis=1), df[label_column]
    data[:] = scale(data[:])

    features = data.columns.values
    w = np.zeros(len(features))

    neighbors = {l:
        {'hit': np.array(data[:][df[label_column] == l]),
        'miss': np.array(data[:][df[label_column] != l])}
        for l in label.unique()
    }

    for _ in range(n_iterations):
        index = np.random.choice(data.index.values)
        x = data.loc[index].values
        y = label.loc[index]
        nearmiss = find_nearest(x, neighbors[y]['miss'])
        nearhit = find_nearest(x, neighbors[y]['hit'])
        for f in range(len(features)):
            w[f] = w[f] + (x[f] - nearmiss[f])**2 - (x[f] - nearhit[f])**2

    w /= n_iterations
    return [features[i] for i in range(len(features)) if w[i] > tau]


def find_nearest(x, X):
    return min(X, key=lambda y: np.linalg.norm(np.subtract(x, y)))


def scale(X):
    scaler = MinMaxScaler(feature_range=(0,1), copy=True)
    return scaler.fit_transform(X)

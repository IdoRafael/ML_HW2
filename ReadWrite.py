import pandas as pd


FILES_DIR = 'CSVFiles\\'


def read_data(online):
    if online:
        return pd.read_csv(
            'https://webcourse.cs.technion.ac.il/236756/Spring2018/ho/WCFiles/ElectionsData.csv?7959',
            header=0
        )
    else:
        return pd.read_csv(FILES_DIR + 'ElectionsData.csv', header=0)


def save_as_csv_original(train, validate, test):
    train.to_csv(FILES_DIR + "train_original.csv", index=False)
    validate.to_csv(FILES_DIR + "validate_original.csv", index=False)
    test.to_csv(FILES_DIR + "test_original.csv", index=False)


def save_as_csv(train, validate, test):
    # TODO ask about filenames expected
    train.to_csv(FILES_DIR + "train.csv", index=False)
    validate.to_csv(FILES_DIR + "validate.csv", index=False)
    test.to_csv(FILES_DIR + "test.csv", index=False)
import pandas as pd


def tvt_split(dataframe, class_col: str, frac: tuple = (70, 15, 15)):
    train_idx = []
    validation_idx = []
    test_idx = []

    if sum(frac) == 100:
        for cls in dataframe[class_col].unique():
            class_len = len(dataframe[dataframe[class_col] == cls])
            tr_am, val_am = int(class_len*frac[0] / 100), int(class_len*frac[1] / 100)
            train = dataframe[dataframe[class_col] == cls].sample(n=tr_am)
            dataframe = dataframe.drop(train.index)
            valid = dataframe[dataframe[class_col] == cls].sample(n=val_am)
            dataframe = dataframe.drop(valid.index)
            test = dataframe[dataframe[class_col] == cls]
            train_idx.extend(train.index)
            validation_idx.extend(valid.index)
            test_idx.extend(test.index)
    else:
        print('Incorrect fraction proportions!')
    return train_idx, validation_idx, test_idx

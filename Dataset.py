import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

import tensorflow_datasets as tfds
import tensorflow as tf

def get_default_preprocessor():
    return make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include=object)),
    )

def identity(x):
    return x

class Dataset:
    def __init__(self, data, y_field=[], y_size=1, y_transform=identity,
                 feature_blacklist=[], x_transform=identity,
                 makePreprocessor=get_default_preprocessor, input_shape=None,
                 batchSize=64, epochs=100, dsType='normal',
                 problemType='regression'):
        if dsType == 'tfds':
            (tfdsData,), tfdsInfo = tfds.load(
                data, split=['all'],
                as_supervised=True,
                data_dir='./datasets/tfds/',
                with_info=True
            )
            data = tfds.as_dataframe(tfdsData, tfdsInfo)

        self.fullData = data
        X = data.copy().dropna().drop(feature_blacklist, axis=1)
        y = X.pop(y_field)

        y = y_transform(y)
        X = x_transform(X)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.70, random_state=42)

        if makePreprocessor != None:
            preprocessor = makePreprocessor()
            X_train = preprocessor.fit_transform(X_train)
            X_valid = preprocessor.transform(X_valid)

        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.50, random_state=42)

        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
            X_test = X_test.toarray()
            X_valid = X_valid.toarray()

        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test

        self.ds_valid = (X_valid, y_valid)

        self.featureCount = self.X_train.shape[1]

        self.y_size = y_size
        self.batchSize = batchSize
        self.epochs = epochs
        self.problemType = problemType

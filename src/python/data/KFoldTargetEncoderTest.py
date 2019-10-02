from sklearn import base

from data.KFoldTargetEncoderTrain import KFOLD_TARGET_ENC_COL_POSTFIX


class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, train, colNames):
        self.train = train
        self.colNames = colNames

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.colNames:
            encoded_name = col+KFOLD_TARGET_ENC_COL_POSTFIX
            mean = self.train[[col, encoded_name]].groupby(col).mean().reset_index()

            dd = {}
            for index, row in mean.iterrows():
                dd[row[col]] = row[encoded_name]
            X[encoded_name] = X[col]
            X = X.replace({encoded_name: dd})
        return X

import logging

import numpy as np
from sklearn import base
from sklearn.model_selection import KFold

KFOLD_TARGET_ENC_COL_POSTFIX = '_Kfold_Target_Enc'


class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):
    logger = logging.getLogger(__name__)

    def __init__(self, colnames, targetName, n_fold=5, verbosity=True, discardOriginal_col=False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert (type(self.targetName) == str)
        assert (self.targetName in X.columns)
        assert (type(self.colnames) == list)
        for col in self.colnames:
            assert (col in X.columns)

        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits=self.n_fold, shuffle=False, random_state=2019)
        for tr_ind, val_ind in kf.split(X):
            for col in self.colnames:
                col_mean_name = col + KFOLD_TARGET_ENC_COL_POSTFIX
                X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
                X.loc[X.index[val_ind], col_mean_name] = X_val[col].map(X_tr.groupby(col)[self.targetName].mean())
                X[col_mean_name].fillna(mean_of_target, inplace=True)


        if self.verbosity:
            for col in self.colnames:
                col_mean_name = col + KFOLD_TARGET_ENC_COL_POSTFIX
                encoded_feature = X[col_mean_name].values
                corr = np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]
                self.logger.debug( "Correlation between the new feature, %s and, %s is %s."%(col, self.targetName, corr))

        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X

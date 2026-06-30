import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            low_limit, up_limit = self.outlier_thresholds(X, variable)
            X.loc[(X[variable] > up_limit), variable] = up_limit
            if X[variable].min() > 0:
                X.loc[(X[variable] < low_limit), variable] = low_limit
        return X

    def outlier_thresholds(self, X, variable, q1_thr=0.25, q3_thr=0.75):
        quartile1 = X[variable].quantile(q1_thr)
        quartile3 = X[variable].quantile(q3_thr)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        age_bins = [-np.inf, 18, 40, 60, np.inf]
        age_labels = [0, 1, 2, 3]
        X['age_category'] = pd.cut(
            X['age'],
            bins=age_bins,
            labels=age_labels
        )

        mhr_bins = [-np.inf, 100, 200, 300, np.inf]
        mhr_labels = [0, 1, 2, 3]
        X['serum_cholesterol_category'] = pd.cut(X['serum_cholesterol'], bins=mhr_bins, labels=mhr_labels)

        X['bp_cholesterol_ratio'] = X['resting_blood_pressure'] / X['serum_cholesterol']

        return X


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("X, must be Pandas DataFrame.")

        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("X, must be Pandas DataFrame.")

        return X.drop(columns=self.to_drop, errors='ignore')



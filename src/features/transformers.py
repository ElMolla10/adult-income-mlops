import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NoneToNaNTransformer(BaseEstimator, TransformerMixin):
    """Convert Python None values to NaN before sklearn imputation."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).replace({None: np.nan})

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder, StandardScaler
from typing import List, Tuple, Union

class PipelineImputer(BaseEstimator, TransformerMixin):
    """
    PipelineImputer fills missing values for numeric and categorical columns.

    Parameters
    ----------
    columns : list or None
        List of columns to impute. If None or empty, all columns are considered.
    numeric_strategy : str
        Strategy for numeric columns: 'mean' | 'median' | 'constant'
    categorical_strategy : str
        Strategy for categorical columns: 'mode' | 'constant'
    fill_value : any
        Value to use when strategy is 'constant' or when aggregation fails.
    """

    def __init__(self, columns: List[str] = None, numeric_strategy: str = "mean", categorical_strategy: str = "mode", fill_value: any = "Unknown"):
        self.columns = columns or []
        self.numeric_strategy = (numeric_strategy or "mean").lower()
        self.categorical_strategy = (categorical_strategy or "mode").lower()
        self.fill_value = fill_value
        self._impute_values = {}

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineImputer':
        if X is None:
            self._impute_values = {}
            return self
        # If no columns were specified, treat this step as a no-op (skip)
        cols = [c for c in (self.columns or []) if c in X.columns]
        if not cols:
            self._impute_values = {}
            return self

        numeric_cols = X[cols].select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in cols if c not in numeric_cols]

        # Compute imputation values
        for c in numeric_cols:
            try:
                if self.numeric_strategy == "mean":
                    val = X[c].mean()
                elif self.numeric_strategy == "median":
                    val = X[c].median()
                else:
                    val = self.fill_value
                if pd.isna(val):
                    val = self.fill_value
            except Exception:
                val = self.fill_value
            self._impute_values[c] = val

        for c in cat_cols:
            try:
                if self.categorical_strategy == "mode":
                    mode_series = X[c].mode()
                    val = mode_series.iloc[0] if not mode_series.empty else self.fill_value
                else:
                    val = self.fill_value
                if pd.isna(val):
                    val = self.fill_value
            except Exception:
                val = self.fill_value
            self._impute_values[c] = val

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None for PipelineImputer.transform")
        X_out = X.copy()
        if not hasattr(self, "_impute_values") or not self._impute_values:
            return X_out
        # Only apply to columns that exist
        for c, val in self._impute_values.items():
            if c in X_out.columns:
                X_out[c] = X_out[c].fillna(val)
        return X_out

class PipelineOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    PipelineOneHotEncoder is a custom transformer that encodes categorical columns using one-hot encoding.
    It is a wrapper around the OneHotEncoder from the sklearn library.
    It takes a list of columns to encode as a parameter.

    Parameters
    ----------
    columns : list
        A list of column names to encode.

    Returns
    -------
    DataFrame
        A new DataFrame with the encoded columns
    """

    def __init__(self, columns: List[str] = None):
        self.columns = columns or []
        if self.columns:
            # scikit-learn >=1.2 renamed `sparse` -> `sparse_output`; try the new name
            try:
                self.encoder = SklearnOneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            except TypeError:
                # fallback for older scikit-learn versions that expect `sparse`
                self.encoder = SklearnOneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
        else:
            self.encoder = None

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineOneHotEncoder':
        if X is None or not self.columns:
            return self
        existing = [c for c in self.columns if c in X.columns]
        if not existing:
            return self
        # fit encoder on only the existing columns
        self.encoder.fit(X[existing])
        self._fitted_columns = existing
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None for PipelineOneHotEncoder.transform")
        X_out = X.copy()
        if not self.columns or self.encoder is None or not hasattr(self, "_fitted_columns"):
            return X_out
        existing = [c for c in self._fitted_columns if c in X_out.columns]
        if not existing:
            return X_out
        encoded = self.encoder.transform(X_out[existing])
        encoded_df = DataFrame(encoded, columns=self.encoder.get_feature_names_out(existing), index=X_out.index)
        X_out = X_out.drop(existing, axis=1, errors="ignore")
        return pd.concat([X_out, encoded_df], axis=1)

class PipelineFeatureDropper(BaseEstimator, TransformerMixin):
    """
    PipelineFeatureDropper is a custom transformer that drops columns from a DataFrame.
    It takes a list of columns to drop as a parameter.

    Parameters
    ----------
    columns : list
        A list of column names to drop.

    Returns
    -------
    DataFrame
        A new DataFrame with the specified columns dropped.
    """

    def __init__(self, columns: List[str] = None):
        '''
        columns : list
            A list of column names to drop.
        '''
        self.columns = columns

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineFeatureDropper':
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None for PipelineFeatureDropper.transform")
        return X.drop(self.columns or [], axis=1, errors="ignore")

class PipelineFeatureStandardScaler(BaseEstimator, TransformerMixin):
    """
    PipelineFeatureStandardScaler is a custom transformer that scales numerical columns using the StandardScaler from the sklearn library.
    It takes a list of columns to scale as a parameter.

    Parameters
    ----------
    columns : list
        A list of column names to scale.

    Returns
    -------
    DataFrame
        A new DataFrame with the specified columns scaled.
    """

    def __init__(self, columns: List[str] = None):
        self.columns = columns or []
        self.scaler = StandardScaler()

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineFeatureStandardScaler':
        if X is None or not self.columns:
            return self
        existing = [c for c in self.columns if c in X.columns]
        if not existing:
            return self
        self.scaler.fit(X[existing])
        self._fitted_columns = existing
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None for PipelineFeatureStandardScaler.transform")
        X_out = X.copy()
        if not hasattr(self, "_fitted_columns") or not self._fitted_columns:
            return X_out
        existing = [c for c in self._fitted_columns if c in X_out.columns]
        if not existing:
            return X_out
        X_out[existing] = self.scaler.transform(X_out[existing])
        return X_out

class PipelineDataSplit(BaseEstimator, TransformerMixin):
    """
    DataSplit is a custom transformer that splits a DataFrame into training and testing sets.

    Parameters
    ----------
    labelColumns : list
        A list of column names to use as labels.
    testSize : float
        The proportion of the dataset to include in the test split.
    randomState : int
        The seed used by the random number generator.

    Returns
    -------
    tuple
        A tuple containing the training and testing sets.
    """

    def __init__(self, labelColumns: List[str] = None, testSize: float = 0.2, randomState: int = 42):
        self.labelColumns = labelColumns or []
        self.testSize = testSize
        self.randomState = randomState

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineDataSplit':
        return self
    
    def transform(self, X: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        if X is None:
            raise ValueError("Input X cannot be None for PipelineDataSplit.transform")
        if not self.labelColumns:
            raise ValueError("labelColumns must be provided for PipelineDataSplit")
        missing = [c for c in self.labelColumns if c not in X.columns]
        if missing:
            raise ValueError(f"Label columns missing from input: {missing}")
        y = X[self.labelColumns]
        X_out = X.drop(self.labelColumns, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X_out, y, test_size=self.testSize, random_state=self.randomState)
        return X_train, X_test, y_train, y_test

class DatasplitToNumpyArray(BaseEstimator, TransformerMixin):
    """
    DatasplitToNumpyArray is a custom transformer that converts the data split into numpy arrays.
    It takes a tuple of DataFrames as input and returns a tuple of numpy arrays.

    Parameters
    ----------
    X : tuple
        A tuple of DataFrames containing the training and testing data.
    y : DataFrame
        A DataFrame containing the target data.

    Returns
    -------
    tuple
        A tuple of numpy arrays containing the training and testing data.
    """

    def fit(self, X: Tuple[DataFrame, DataFrame, DataFrame, DataFrame], y: Union[DataFrame, None] = None) -> 'DatasplitToNumpyArray':
        return self
    
    def transform(self, X: Tuple[DataFrame, DataFrame, DataFrame, DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if X is None:
            raise ValueError("Input X cannot be None for DatasplitToNumpyArray.transform")
        if not isinstance(X, (list, tuple)) or len(X) != 4:
            raise ValueError("DatasplitToNumpyArray.transform expects a tuple/list of four DataFrames: X_train, X_test, y_train, y_test")
        X_train, X_test, y_train, y_test = X
        return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

class PipelineDateSpliter(BaseEstimator, TransformerMixin):
    """
    PipelineDateSpliter is a custom transformer that splits date columns into day, month, and year columns.
    It takes a list of columns to split as a parameter.

    Parameters
    ----------
    columns : list
        A list of column names to split.

    Returns
    -------
    DataFrame
        A new DataFrame with the specified columns split.
    """

    def __init__(self, columns: List[str] = None, dropColumns: bool = True):
        self.columns = columns or []
        self.dropColumns = dropColumns

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineDateSpliter':
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None for PipelineDateSpliter.transform")
        X_out = X.copy()
        # No-op if no columns specified
        if not self.columns:
            return X_out
        for column in self.columns:
            if column not in X_out.columns:
                continue
            X_out[column] = pd.to_datetime(X_out[column], errors='coerce')
            X_out[column + "_year"] = X_out[column].dt.year
            X_out[column + "_month"] = X_out[column].dt.month
            X_out[column + "_day"] = X_out[column].dt.day
            if self.dropColumns:
                X_out = X_out.drop([column], axis=1, errors="ignore")
        return X_out

class PipelineIndexSetter(BaseEstimator, TransformerMixin):
    """
    PipelineIndexSetter is a custom transformer that sets the index of a DataFrame.
    It takes a column name to use as the index as a parameter.

    Parameters
    ----------
    index : str
        A column name to use as the index.

    Returns
    -------
    DataFrame
        A new DataFrame with the specified column as the index.
    """

    def __init__(self, index: str = None):
        self.index = index

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineIndexSetter':
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None for PipelineIndexSetter.transform")
        X_out = X.copy()
        # No-op if index not specified or not in columns
        if not self.index or not isinstance(self.index, str) or self.index.strip() == "":
            return X_out
        if self.index not in X_out.columns:
            return X_out
        return X_out.set_index(self.index)

class PipelineSequencer(BaseEstimator, TransformerMixin):
    """
    PipelineSequencer is a custom transformer that creates sequences from a DataFrame.
    It takes a column to use as the label and a sequence length as parameters.

    Parameters
    ----------
    column : str
        A column name to use as the label.
    sequence_length : int
        The length of the sequence to create.

    Returns
    -------
    tuple
        A tuple containing a list of DataFrames and a DataFrame.
    """

    def __init__(self, column: str = None, sequence_length: int = 60):
        self.column = column
        self.sequence_length = sequence_length

    def fit(self, X: DataFrame, y: Union[DataFrame, None] = None) -> 'PipelineSequencer':
        return self
    
    def transform(self, X: DataFrame) -> DataFrame:
        if X is None:
            raise ValueError("Input X cannot be None for PipelineSequencer.transform")
        # No-op if column not specified or not in dataframe
        if not self.column or not isinstance(self.column, str) or self.column.strip() == "":
            return X.copy()
        if self.column not in X.columns:
            return X.copy()
        # Only create sequences if properly configured
        sequences, labels = self.create_sequences(X)
        # For now, just return the input as-is since sequences return incompatible format
        # This transformer needs broader refactoring for sklearn compatibility
        return X.copy()
        
    def create_sequences(self, data: DataFrame) -> Tuple[List[DataFrame], DataFrame]:
        sequencedX = []
        sequencedy = []
        for i in range(self.sequence_length, len(data)):
            sequence_df = data.iloc[i-self.sequence_length:i].copy()
            sequencedX.append(sequence_df)
            sequencedy.append(data.iloc[i][self.column])
                        
        label_df = pd.DataFrame(sequencedy, columns=['label'])
        return sequencedX, label_df
"""
preprocessing.py

This module provides a collection of data preprocessing functions for use in data analysis and machine learning pipelines.
It includes utilities for encoding categorical variables, scaling numerical features, handling missing values, and managing outliers.
The functions are designed to work with pandas DataFrames and leverage popular libraries such as scikit-learn and scipy.
made by: [Moaz Hany]
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats

def label_encoder(df, columns):
    """Apply Label Encoding to specified columns."""
    df = df.copy()
    le = LabelEncoder()
    for col in columns:
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
            raise ValueError(f"Label Encoding can only be applied to categorical columns. Column '{col}' is not categorical.")
        if df[col].isna().any():
            df[col] = df[col].fillna('MISSING')
        df[col] = le.fit_transform(df[col])
    return df

def one_hot_encoder(df, columns, max_categories=10):
    """Apply One-Hot Encoding to specified columns with category limit."""
    df = df.copy()
    for col in columns:
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
            raise ValueError(f"One-Hot Encoding can only be applied to categorical columns. Column '{col}' is not categorical.")
        
        value_counts = df[col].value_counts()
        if len(value_counts) > max_categories:
            top_categories = value_counts.index[:max_categories-1]
            df[col] = df[col].where(df[col].isin(top_categories), 'OTHER')
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    transformed = ohe.fit_transform(df[columns])
    ohe_df = pd.DataFrame(transformed, columns=ohe.get_feature_names_out(columns), index=df.index)
    df = df.drop(columns=columns, axis=1)
    df = pd.concat([df, ohe_df], axis=1)
    return df

def standard_scaler(df, columns):
    """Apply Standard Scaling to specified columns."""
    df = df.copy()
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Standard Scaling can only be applied to numeric columns. Column '{col}' is not numeric.")
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def minmax_scaler(df, columns):
    """Apply Min-Max Scaling to specified columns."""
    df = df.copy()
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Min-Max Scaling can only be applied to numeric columns. Column '{col}' is not numeric.")
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def handle_missing_values(df, columns, method, strategy='mean', fill_value=None, **kwargs):
    """Handle missing values with various methods."""
    df = df.copy()
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
    
    empty_cols = [col for col in columns if df[col].isna().all()]
    if empty_cols:
        raise ValueError(f"Columns {empty_cols} are completely empty. Consider dropping them.")
    
    if method == 'simple':
        if strategy in ['mean', 'median'] and not numeric_cols:
            raise ValueError(f"Cannot use strategy '{strategy}' with non-numeric columns.")
        
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        if numeric_cols:
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent', fill_value=fill_value)
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            
    elif method == 'knn':
        if not numeric_cols:
            raise ValueError("KNN imputation requires at least one numeric column.")
        imputer = KNNImputer(**kwargs)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
    elif method == 'iterative':
        if not numeric_cols:
            raise ValueError("Iterative imputation requires at least one numeric column.")
        imputer = IterativeImputer(**kwargs)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
    else:
        raise ValueError(f"Invalid imputation method '{method}'. Choose 'simple', 'knn', or 'iterative'.")
    
    return df

def remove_outliers_zscore(df, columns, threshold=3):
    """Remove outliers using Z-score method."""
    df = df.copy()
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Z-score outlier detection can only be applied to numeric columns. Column '{col}' is not numeric.")
        z_scores = np.abs(stats.zscore(df[col]))
        df = df[z_scores < threshold]
    return df.reset_index(drop=True)

def winsorize(df, columns, lower_quantile=0.05, upper_quantile=0.95):
    """Apply winsorization to limit extreme values."""
    df = df.copy()
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Winsorization can only be applied to numeric columns. Column '{col}' is not numeric.")
        lower = df[col].quantile(lower_quantile)
        upper = df[col].quantile(upper_quantile)
        df[col] = np.clip(df[col], lower, upper)
    return df

def clipping(df, columns, min_value=None, max_value=None):
    """Clip values outside of [min_value, max_value]."""
    df = df.copy()
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Clipping can only be applied to numeric columns. Column '{col}' is not numeric.")
        df[col] = df[col].clip(lower=min_value, upper=max_value)
    return df
"""
utils.py

This module provides utility functions for dataset analysis, preprocessing recommendations, model suggestions, and data cleaning.
It helps in understanding the structure of a pandas DataFrame, identifying missing values, suggesting target variables and problem types,
and generating recommendations for preprocessing and modeling. It also includes a function to clean raw data for further analysis.
made by: [Moaz Hany]
"""

import pandas as pd
import numpy as np
import streamlit as st

def analyze_dataset(df):
    """Analyze the dataset and provide recommendations."""
    analysis = {
        'missing_values': {},
        'categorical_features': [],
        'numerical_features': [],
        'target_suggestion': None,
        'problem_type': None
    }
    
    # Basic analysis
    for col in df.columns:
        # Missing values analysis
        missing_percent = df[col].isna().mean() * 100
        if missing_percent > 0:
            analysis['missing_values'][col] = missing_percent
        
        # Data type analysis
        if pd.api.types.is_numeric_dtype(df[col]):
            analysis['numerical_features'].append(col)
        else:
            analysis['categorical_features'].append(col)
    
    # Target variable suggestion
    potential_targets = []
    for col in df.columns:
        if df[col].isna().mean() < 0.3:  # Less than 30% missing
            unique_ratio = df[col].nunique() / len(df)
            if 0 < unique_ratio < 0.5:  # Reasonable cardinality for target
                potential_targets.append((col, unique_ratio))
    
    if potential_targets:
        potential_targets.sort(key=lambda x: abs(x[1] - 0.1))
        analysis['target_suggestion'] = potential_targets[0][0]
        
        unique_values = df[analysis['target_suggestion']].nunique()
        if unique_values <= 10:
            analysis['problem_type'] = 'classification'
        else:
            analysis['problem_type'] = 'regression'
    
    return analysis

def get_preprocessing_recommendations(analysis, df):
    """Generate preprocessing recommendations based on dataset analysis."""
    recommendations = []
    
    # Missing values recommendations
    if analysis['missing_values']:
        rec = "### Missing Values Handling\n"
        for col, percent in analysis['missing_values'].items():
            if percent > 30:
                rec += f"- Column '{col}' has {percent:.1f}% missing values. Consider dropping it.\n"
            elif percent > 5:
                if col in analysis['numerical_features']:
                    rec += f"- Column '{col}' has {percent:.1f}% missing values. Recommend iterative imputation or KNN imputation.\n"
                else:
                    rec += f"- Column '{col}' has {percent:.1f}% missing values. Recommend mode imputation.\n"
            else:
                rec += f"- Column '{col}' has {percent:.1f}% missing values. Recommend simple imputation.\n"
        recommendations.append(rec)
    
    # Encoding recommendations
    if analysis['categorical_features']:
        rec = "### Categorical Encoding\n"
        for col in analysis['categorical_features']:
            unique_count = df[col].nunique()
            if unique_count <= 5:
                rec += f"- Column '{col}' has {unique_count} unique values. Recommend one-hot encoding.\n"
            elif unique_count <= 20:
                rec += f"- Column '{col}' has {unique_count} unique values. Consider label encoding or one-hot encoding with max categories.\n"
            else:
                rec += f"- Column '{col}' has {unique_count} unique values. Recommend label encoding or consider dropping.\n"
        recommendations.append(rec)
    
    # Scaling recommendations
    if analysis['numerical_features']:
        rec = "### Feature Scaling\n"
        rec += "- Numerical features detected. Recommend standard scaling for normally distributed data or min-max scaling otherwise.\n"
        recommendations.append(rec)
    
    return recommendations

def get_model_recommendations(analysis, df):
    """Generate model recommendations based on dataset characteristics."""
    if not analysis['problem_type']:
        return "Cannot recommend models without identifying a suitable target variable."
    
    if analysis['problem_type'] == 'classification':
        rec = "### Classification Model Recommendations\n"
        num_samples = len(df)
        num_features = len(analysis['numerical_features']) + len(analysis['categorical_features'])
        
        rec += "- For most classification problems, Random Forest is a good starting point.\n"
        
        if num_samples < 1000:
            rec += "- With small datasets (<1k samples), consider Logistic Regression or SVM with simple kernels.\n"
        else:
            rec += "- With larger datasets, Gradient Boosting often performs well but may be slower to train.\n"
        
        if num_features > 50:
            rec += "- With many features, consider regularization (Logistic Regression) or feature selection.\n"
        
        unique_classes = df[analysis['target_suggestion']].nunique()
        if unique_classes == 2:
            rec += "- For binary classification, all models are suitable.\n"
        else:
            rec += f"- For multi-class classification ({unique_classes} classes), consider models that handle multi-class well like Random Forest.\n"
        
        return rec
    
    else:  # regression
        rec = "### Regression Model Recommendations\n"
        rec += "- For most regression problems, Gradient Boosting or Random Forest are good starting points.\n"
        rec += "- For linear relationships, consider Linear Regression or Regularized Regression models.\n"
        rec += "- With many features, consider regularization or feature selection.\n"
        return rec

def clean_data(df):
    """Clean the raw dataframe."""
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].replace(
                ['-,---', '--', '---', '-', 'NaN', 'nan', '', 'null', 'NULL', 'None', '?'], np.nan
            )
            try:
                df_cleaned[col] = df_cleaned[col].str.replace(',', '', regex=False)
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')
            except AttributeError:
                pass
    return df_cleaned
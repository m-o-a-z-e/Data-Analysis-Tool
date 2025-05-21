"""
modeling.py

This module provides functions for building, training, evaluating, and visualizing machine learning classification models.
It supports multiple algorithms (Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting, KNN, Naive Bayes),
feature scaling, cross-validation, and performance metrics visualization. The functions are designed to work with pandas DataFrames
and integrate with Streamlit for interactive model selection and evaluation.
made by
data split -- [Hagar Mostafa]
models -- [Nour Mohammed + Moaz Hany]
evaluation -- [Ammar Amgad]
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def get_model(model_option, params):
    """Return the appropriate model based on user selection."""
    if model_option == "Logistic Regression":
        return LogisticRegression(C=params.get('C', 1.0), max_iter=params.get('max_iter', 1000))
    elif model_option == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params.get('max_depth', 5),
            min_samples_split=params.get('min_samples_split', 2)
        )
    elif model_option == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None)
        )
    elif model_option == "SVM":
        return SVC(
            kernel=params.get('kernel', 'rbf'),
            C=params.get('C', 1.0)
        )
    elif model_option == "Gradient Boosting":
        return GradientBoostingClassifier(
            learning_rate=params.get('learning_rate', 0.1),
            n_estimators=params.get('n_estimators', 100)
        )
    elif model_option == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=params.get('n_neighbors', 5))
    elif model_option == "Naive Bayes":
        return GaussianNB()
    else:
        raise ValueError(f"Unknown model option: {model_option}")

def evaluate_model(model, X_train, X_test, y_train, y_test, use_kfold=False, n_splits=5, shuffle_data=False, random_state=42):
    """Evaluate the model and return results."""
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Cross-Validation
    if use_kfold:
        kf = KFold(n_splits=n_splits, shuffle=shuffle_data, random_state=random_state)
        cv_scores = cross_val_score(model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), 
                                  cv=kf, scoring='accuracy')
        metrics['cv_scores'] = cv_scores
    else:
        metrics['cv_scores'] = None
    
    return metrics

def plot_confusion_matrix(cm):
    """Plot confusion matrix."""
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

def run_model_evaluation(df, target_column, model_option, model_params, 
                        test_size=0.2, random_state=42, scale_features=False, 
                        scaler_type="Standard", use_kfold=False, n_splits=5, 
                        shuffle_data=False):
    """Run complete model evaluation pipeline."""
    try:
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode target if categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Handle categorical features
        X = pd.get_dummies(X, drop_first=True)
        
        # Feature scaling
        if scale_features:
            if scaler_type == "Standard":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if use_kfold else None
        )
        
        # Get model
        model = get_model(model_option, model_params)
        
        # Evaluate model
        metrics = evaluate_model(
            model, X_train, X_test, y_train, y_test, 
            use_kfold=use_kfold, n_splits=n_splits, 
            shuffle_data=shuffle_data, random_state=random_state
        )
        
        # Prepare classification report DataFrame
        classification_report_df = pd.DataFrame(metrics['classification_report']).transpose()
        
        return {
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "classification_report_df": classification_report_df,
            "confusion_matrix": metrics['confusion_matrix'],
            "cv_scores": metrics['cv_scores']
        }
        
    except Exception as e:
        st.error(f"Error during model evaluation: {str(e)}")
        return None
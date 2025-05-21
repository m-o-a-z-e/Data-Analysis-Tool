"""
main.py

This is the main entry point for the Advanced Data Preprocessing and Modeling GUI application.
It provides an interactive Streamlit interface for uploading datasets, performing preprocessing (encoding, missing value handling, outlier treatment),
visualizing data, and building/evaluating machine learning models. The app leverages modular functions for preprocessing, visualization, modeling,
and utility recommendations to guide users through the data science workflow.
made by: [ِAbdelrahman Salah revwewed and modified by Moaz Hany]
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from preprocessing import *
from visualization import *
from modeling import *
from utils import *

def main():
    st.set_page_config(layout="wide")
    st.title("Data Analysis Tool")
    
    # Add warning about large datasets
    st.sidebar.warning("For datasets >100MB, consider sampling for better performance.")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Original Data", df.head())
            
            # =============== Smart Recommendations ===============
            st.header("Smart Recommendations")
            
            with st.expander("Show Dataset Analysis & Recommendations"):
                analysis = analyze_dataset(df)
                
                st.subheader("Dataset Analysis")
                st.write(f"- Total samples: {len(df)}")
                st.write(f"- Total features: {len(df.columns)}")
                st.write(f"- Numerical features: {len(analysis['numerical_features'])}")
                st.write(f"- Categorical features: {len(analysis['categorical_features'])}")
                st.write(f"- Features with missing values: {len(analysis['missing_values'])}")
                
                if analysis['target_suggestion']:
                    st.write(f"- Suggested target variable: '{analysis['target_suggestion']}'")
                    st.write(f"- Problem type: {analysis['problem_type']}")
                else:
                    st.warning("No clear target variable identified. Please manually select one.")
                
                st.subheader("Preprocessing Recommendations")
                preprocessing_recs = get_preprocessing_recommendations(analysis, df)
                if preprocessing_recs:
                    for rec in preprocessing_recs:
                        st.markdown(rec)
                else:
                    st.write("No specific preprocessing recommendations.")
                
                st.subheader("Modeling Recommendations")
                st.markdown(get_model_recommendations(analysis, df))
            
            # Add option to sample data
            if len(df) > 10000:
                sample_size = st.slider("Sample size (for large datasets)", 1000, 10000, 5000)
                df = shuffle(df).sample(sample_size).reset_index(drop=True)
                st.info(f"Using sampled dataset with {len(df)} rows for better performance.")
            
            # Data cleaning
            df_cleaned = clean_data(df)
            
            # Add option to drop columns with too many missing values
            missing_threshold = st.slider("Drop columns with missing values > (%)", 0, 100, 50)
            missing_percent = df_cleaned.isnull().mean() * 100
            cols_to_drop = missing_percent[missing_percent > missing_threshold].index.tolist()
            
            if cols_to_drop:
                st.warning(f"Columns with >{missing_threshold}% missing values: {cols_to_drop}")
                if st.checkbox(f"Drop these {len(cols_to_drop)} columns?"):
                    df_cleaned = df_cleaned.drop(columns=cols_to_drop)
                    st.success(f"Dropped columns: {cols_to_drop}")
            
            df_copy = df_cleaned.copy()
            
            # =================== Encoding ===================
            st.sidebar.header("Encoding Options")
            categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                label_cols = st.sidebar.multiselect("Columns for Label Encoding", categorical_cols)
                onehot_cols = st.sidebar.multiselect("Columns for One-Hot Encoding", categorical_cols)
                
                if label_cols and onehot_cols and set(label_cols).intersection(set(onehot_cols)):
                    st.error("A column cannot have both Label Encoding and One-Hot Encoding!")
                else:
                    if label_cols:
                        df_copy = label_encoder(df_copy, label_cols)
                        st.success("Label Encoding applied!")
                    if onehot_cols:
                        max_categories = st.sidebar.number_input("Max categories for One-Hot (to prevent explosion)", 
                                                                min_value=2, max_value=50, value=10)
                        df_copy = one_hot_encoder(df_copy, onehot_cols, max_categories)
                        st.success("One-Hot Encoding applied!")
            else:
                st.sidebar.info("No categorical columns found for encoding")
            
            # =================== Missing Values Handling ===================
            st.sidebar.header("Missing Values Handling")
            missing_cols = df_copy.columns[df_copy.isna().any()].tolist()
            
            if missing_cols:
                st.sidebar.subheader("Available columns with missing values")
                st.sidebar.write(missing_cols)
                
                method = st.sidebar.selectbox("Imputation Method", 
                                             ["simple", "knn", "iterative", "drop rows", "drop columns"])
                
                if method in ["simple", "knn", "iterative"]:
                    selected_cols = st.sidebar.multiselect("Select columns to impute", missing_cols)
                    
                    if method == "simple":
                        strategy = st.sidebar.selectbox("Simple strategy", ["mean", "median", "most_frequent", "constant"])
                        fill_value = None
                        if strategy == "constant":
                            fill_value = st.sidebar.text_input("Fill value", value="0")
                            try:
                                fill_value = float(fill_value) if '.' in fill_value else int(fill_value)
                            except ValueError:
                                pass
                        
                        if selected_cols:
                            df_copy = handle_missing_values(df_copy, selected_cols, method='simple', 
                                                          strategy=strategy, fill_value=fill_value)
                            st.success(f"Simple imputation ({strategy}) applied!")
                    
                    elif method == "knn":
                        n_neighbors = st.sidebar.number_input("Number of neighbors", min_value=1, max_value=10, value=5)
                        if selected_cols:
                            df_copy = handle_missing_values(df_copy, selected_cols, method='knn', n_neighbors=n_neighbors)
                            st.success("KNN imputation applied!")
                    
                    elif method == "iterative":
                        max_iter = st.sidebar.number_input("Maximum iterations", min_value=1, max_value=20, value=10)
                        if selected_cols:
                            df_copy = handle_missing_values(df_copy, selected_cols, method='iterative', max_iter=max_iter)
                            st.success("Iterative imputation applied!")
                
                elif method == "drop rows":
                    if st.sidebar.button("Drop rows with missing values"):
                        initial_rows = len(df_copy)
                        df_copy = df_copy.dropna()
                        final_rows = len(df_copy)
                        st.success(f"Dropped {initial_rows - final_rows} rows with missing values")
                
                elif method == "drop columns":
                    if st.sidebar.button("Drop columns with missing values"):
                        initial_cols = len(df_copy.columns)
                        df_copy = df_copy.dropna(axis=1)
                        final_cols = len(df_copy.columns)
                        st.success(f"Dropped {initial_cols - final_cols} columns with missing values")
            else:
                st.sidebar.info("No missing values found in the dataset")
            
            # =================== Outlier Handling ===================
            st.sidebar.header("Outlier Handling")
            numeric_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_cols:
                outlier_method = st.sidebar.selectbox("Outlier Treatment Method", 
                                                    ["None", "Z-score", "Winsorization", "Clipping"])
                
                if outlier_method == "Z-score":
                    zscore_cols = st.sidebar.multiselect("Select numeric columns", numeric_cols)
                    zscore_threshold = st.sidebar.slider("Z-score threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
                    if zscore_cols:
                        initial_rows = len(df_copy)
                        df_copy = remove_outliers_zscore(df_copy, zscore_cols, threshold=zscore_threshold)
                        final_rows = len(df_copy)
                        st.success(f"Removed {initial_rows - final_rows} outliers using Z-score")
                
                elif outlier_method == "Winsorization":
                    winsorize_cols = st.sidebar.multiselect("Select numeric columns", numeric_cols)
                    lower_q = st.sidebar.slider("Lower quantile", 0.0, 0.2, 0.05, 0.01)
                    upper_q = st.sidebar.slider("Upper quantile", 0.8, 1.0, 0.95, 0.01)
                    if winsorize_cols:
                        df_copy = winsorize(df_copy, winsorize_cols, lower_q, upper_q)
                        st.success("Winsorization applied")
                
                elif outlier_method == "Clipping":
                    clip_cols = st.sidebar.multiselect("Select numeric columns", numeric_cols)
                    if clip_cols:
                        col_min = df_copy[clip_cols].min().min()
                        col_max = df_copy[clip_cols].max().max()
                        min_val = st.sidebar.number_input("Minimum value", value=col_min)
                        max_val = st.sidebar.number_input("Maximum value", value=col_max)
                        if st.sidebar.button("Apply Clipping"):
                            df_copy = clipping(df_copy, clip_cols, min_val, max_val)
                            st.success("Clipping applied")
            else:
                st.sidebar.info("No numeric columns found for outlier handling")
            
            # ===================== Display the Processed Data =====================
            st.write("### Processed Data", df_copy.head())
            
            # Add data summary
            st.subheader("Data Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Shape:**", df_copy.shape)
                st.write("**Missing Values:**")
                st.write(df_copy.isna().sum())
            
            with col2:
                st.write("**Data Types:**")
                st.write(df_copy.dtypes.value_counts())
                st.write("**Numeric Columns Summary:**")
                st.write(df_copy.describe())
            
            # Download button
            st.sidebar.download_button("Download Processed Data", 
                                      data=df_copy.to_csv(index=False), 
                                      file_name="processed_data.csv", 
                                      mime="text/csv")
            
            # =============== Visualization Section ===============
            st.header("Data Visualization")
            data_option = st.radio("Select data version to visualize:", ["Before Preprocessing", "After Preprocessing"])
            selected_df = df_cleaned if data_option == "Before Preprocessing" else df_copy
            
            visualize_data(selected_df)
            
            # =================== Target Selection ===================
            st.sidebar.header("Modeling Options")
            target_column = st.sidebar.selectbox("Choose the target column", df_copy.columns)
            
            if target_column:
                st.sidebar.write(f"Selected target column: {target_column}")
                
                # =============== Model Selection ===============
                model_option = st.sidebar.selectbox("Select a model", 
                                                    ["Logistic Regression", "Decision Tree", "Random Forest", 
                                                     "Gradient Boosting", "K-Nearest Neighbors", "Naive Bayes"])
                
                # Model parameters
                model_params = {}
                if model_option == "Logistic Regression":
                    model_params['C'] = st.sidebar.number_input("Regularization strength (C)", 0.01, 10.0, 1.0)
                elif model_option == "Decision Tree":
                    model_params['max_depth'] = st.sidebar.number_input("Max depth", 1, 20, 5)
                elif model_option == "Random Forest":
                    model_params['n_estimators'] = st.sidebar.number_input("Number of trees", 1, 100, 10)
                elif model_option == "Gradient Boosting":
                    model_params['learning_rate'] = st.sidebar.number_input("Learning rate", 0.01, 1.0, 0.1)
                    model_params['n_estimators'] = st.sidebar.number_input("Number of trees", 1, 100, 100)
                elif model_option == "K-Nearest Neighbors":
                    model_params['n_neighbors'] = st.sidebar.number_input("Number of neighbors", 1, 20, 5)
                
                # Train-Test Split
                test_size = st.sidebar.slider("Test size (%)", 10, 50, 20) / 100
                random_state = st.sidebar.number_input("Random state", value=42)
                
                # K-Fold Cross Validation
                use_kfold = st.sidebar.checkbox("Use K-Fold Cross Validation")
                n_splits = st.sidebar.number_input("Number of splits for K-Fold", min_value=2, max_value=10, value=5)
                
                # Shuffle data
                shuffle_data = st.sidebar.checkbox("Shuffle data before splitting")
                
                if st.sidebar.button("Run Model Evaluation"):
                    try:
                        metrics = run_model_evaluation(
                            df_copy, target_column, model_option, 
                            model_params=model_params,
                            test_size=test_size,
                            random_state=random_state,
                            use_kfold=use_kfold,
                            n_splits=n_splits,
                            shuffle_data=shuffle_data
                        )
                        st.subheader("Model Evaluation Metrics")

                        # Improved metrics display
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                            st.metric("Precision (weighted)", f"{metrics.get('precision', 0):.4f}")
                        with col2:
                            st.metric("Recall (weighted)", f"{metrics.get('recall', 0):.4f}")
                            st.metric("F1 Score (weighted)", f"{metrics.get('f1', 0):.4f}")

                        # Classification report
                        if 'classification_report_df' in metrics:
                            st.markdown("#### Classification Report")
                            st.dataframe(metrics['classification_report_df'])

                        # Confusion matrix
                        if 'confusion_matrix' in metrics:
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            st.markdown("#### Confusion Matrix")
                            fig, ax = plt.subplots()
                            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)

                        # Cross-validation results
                        if use_kfold and 'cv_scores' in metrics:
                            st.markdown(f"#### {n_splits}-Fold Cross-Validation Results")
                            st.write(f"Mean Accuracy: {np.mean(metrics['cv_scores']):.4f}")
                            st.write(f"Standard Deviation: {np.std(metrics['cv_scores']):.4f}")

                    except Exception as e:
                        st.error(f"Error during model evaluation: {str(e)}")
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            st.stop()
    else:
        st.info("Please upload a CSV file to get started.")
if __name__ == "__main__":
    main()
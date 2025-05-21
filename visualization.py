"""
visualization.py

This module provides a set of functions for interactive data visualization using Streamlit, Seaborn, and Matplotlib.
It enables users to generate a variety of plots (line, scatter, box, pie, pairplot, heatmap, distribution, count, and bar plots)
for exploratory data analysis. The functions are designed to work with pandas DataFrames and support filtering and customization
for better insights and presentation.
made by: [Abdelrahman Salah]
"""

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_filtered_df(df, cat_col, top_n=10):
    """Filter dataframe to top N categories for visualization."""
    top_values = df[cat_col].value_counts().nlargest(top_n).index
    return df[df[cat_col].isin(top_values)].copy()

def shorten_labels(values, max_len=20):
    """Shorten long labels for visualization."""
    return [str(x)[:max_len] + "..." if len(str(x)) > max_len else str(x) for x in values]

def plot_line(selected_df, num_cols):
    """Generate line plot."""
    if num_cols:
        line_col = st.selectbox("Select column for Line Plot", num_cols)
        st.line_chart(selected_df[line_col])
    else:
        st.warning("No numeric columns available for Line Plot")

def plot_scatter(selected_df, num_cols):
    """Generate scatter plot."""
    if len(num_cols) >= 2:
        x_col = st.selectbox("X-axis", num_cols, key="x_scatter")
        y_col = st.selectbox("Y-axis", num_cols, key="y_scatter")
        fig, ax = plt.subplots()
        sns.scatterplot(data=selected_df, x=x_col, y=y_col, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Need at least 2 numeric columns for Scatter Plot")

def plot_box(selected_df, num_cols):
    """Generate box plot."""
    if num_cols:
        box_col = st.selectbox("Select column for Box Plot", num_cols)
        fig, ax = plt.subplots()
        sns.boxplot(y=selected_df[box_col], ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for Box Plot")

def plot_pie(selected_df, cat_cols):
    """Generate pie chart."""
    if cat_cols:
        pie_col = st.selectbox("Select categorical column for Pie Chart", cat_cols)
        filtered_df = get_filtered_df(selected_df, pie_col)
        pie_data = filtered_df[pie_col].value_counts()
        pie_data.index = shorten_labels(pie_data.index)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.warning("No categorical columns available for Pie Chart")

def plot_pairplot(selected_df, num_cols):
    """Generate pairplot."""
    if len(num_cols) >= 2:
        fig = sns.pairplot(selected_df[num_cols].sample(min(1000, len(selected_df))))
        st.pyplot(fig)
    else:
        st.warning("Need at least 2 numeric columns for Pairplot")

def plot_heatmap(selected_df, num_cols):
    """Generate heatmap."""
    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(selected_df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Need at least 2 numeric columns for Heatmap")

def plot_distribution(selected_df, num_cols):
    """Generate distribution plot."""
    if num_cols:
        dist_col = st.selectbox("Select column for Distribution Plot", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(selected_df[dist_col], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for Distribution Plot")

def plot_count(selected_df, cat_cols):
    """Generate count plot."""
    if cat_cols:
        count_col = st.selectbox("Select categorical column for Count Plot", cat_cols)
        filtered_df = get_filtered_df(selected_df, count_col)
        filtered_df[count_col] = shorten_labels(filtered_df[count_col])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=filtered_df, y=count_col, ax=ax)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        st.pyplot(fig)
    else:
        st.warning("No categorical columns available for Count Plot")

def plot_bar(selected_df, cat_cols, num_cols):
    """Generate bar plot."""
    if cat_cols and num_cols:
        bar_x = st.selectbox("Select X (categorical) column", cat_cols)
        bar_y = st.selectbox("Select Y (numeric) column", num_cols)
        filtered_df = get_filtered_df(selected_df, bar_x)
        filtered_df[bar_x] = shorten_labels(filtered_df[bar_x])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=filtered_df, y=bar_x, x=bar_y, ax=ax)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        st.pyplot(fig)
    else:
        st.warning("Need both categorical and numeric columns for Bar Plot")

def visualize_data(selected_df):
    """Main visualization function."""
    st.header("Data Visualization")
    
    plot_type = st.selectbox(
        "Select Plot Type",
        ["Line Plot", "Scatter Plot", "Box Plot", "Pie Chart", "Pairplot", 
         "Heatmap", "Distribution Plot", "Count Plot", "Bar Plot"]
    )
    
    num_cols = selected_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = selected_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(selected_df) == 0:
        st.warning("The selected dataset is empty after processing.")
    else:
        try:
            if plot_type == "Line Plot":
                plot_line(selected_df, num_cols)
            elif plot_type == "Scatter Plot":
                plot_scatter(selected_df, num_cols)
            elif plot_type == "Box Plot":
                plot_box(selected_df, num_cols)
            elif plot_type == "Pie Chart":
                plot_pie(selected_df, cat_cols)
            elif plot_type == "Pairplot":
                plot_pairplot(selected_df, num_cols)
            elif plot_type == "Heatmap":
                plot_heatmap(selected_df, num_cols)
            elif plot_type == "Distribution Plot":
                plot_distribution(selected_df, num_cols)
            elif plot_type == "Count Plot":
                plot_count(selected_df, cat_cols)
            elif plot_type == "Bar Plot":
                plot_bar(selected_df, cat_cols, num_cols)
        except Exception as e:
            st.error(f"Error creating plot: {str(e)}")
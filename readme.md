# Advanced Data Preprocessing and Modeling GUI

A professional, modular Streamlit application for end-to-end data analysis, preprocessing, visualization, and machine learning modeling. Designed to empower both technical and non-technical users to explore, clean, visualize, and model datasets through an intuitive graphical interface.

---

## Features

- **Intuitive Data Upload:** Upload CSV datasets directly via the web interface.
- **Automated Data Analysis & Recommendations:** Instantly receive a summary of your dataset, including feature types, missing values, suggested target variable, and problem type.
- **Flexible Preprocessing Pipeline:**
  - Encoding: Label Encoding and One-Hot Encoding with category limit.
  - Missing Value Handling: Simple, KNN, Iterative imputation, or dropping rows/columns.
  - Outlier Treatment: Z-score removal, Winsorization, or value clipping.
  - Column Management: Drop columns with excessive missing data.
- **Interactive Visualization Suite:** Generate a wide range of customizable plots (scatter, box, heatmap, pairplot, etc.) before and after preprocessing.
- **Comprehensive Modeling:**
  - Select from Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, and Naive Bayes.
  - Adjustable hyperparameters for each model.
  - Supports train/test split and K-Fold cross-validation.
  - Detailed metrics: accuracy, precision, recall, F1, classification report, and confusion matrix.
- **Export:** Download your processed dataset as a CSV file.

---

## Installation Instructions

1. **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd "DA Project"
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Use

1. **Run the application:**
    ```bash
    streamlit run main.py
    ```
2. **Access the app:**  
   Open the provided local URL (typically [http://localhost:8501](http://localhost:8501)) in your browser.
3. **Upload your CSV file** and follow the sidebar options to preprocess, visualize, and model your data.

---

## Technologies Used

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scipy](https://scipy.org/)

---

## Screenshots

> _Add screenshots of the application here to showcase the interface and features._

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

---

## License

This project is for educational and research purposes.

---

## Contact / Author Info

- **Main App:** Abdelrahman Salah (reviewed and enhanced by Moaz Hany)
- **Preprocessing:** Moaz Hany
- **Visualization:** Abdelrahman Salah
- **Modeling:** Hagar Mostafa, Nour Mohammed, Moaz Hany, Ammar Amgad

For questions or suggestions, please open an issue or contact the maintainers.
# ElevateLabs-Task-06-KNN-Classification

## Overview
This repository contains a complete, production-grade K-Nearest Neighbors (KNN) classification project. It applies the KNN algorithm to classify iris flowers into three distinct species based on their sepal and petal measurements.

## Objective
To build an internship-ready classification project utilizing best engineering practices, demonstrating proper data ingestion (SQL and CSV), preprocessing (StandardScaler and scaling without data leakage), hyperparameter comparison ($K$-value evaluation), decision boundary mapping, and model evaluation metrics.

## Dataset
The project utilizes the classic **Iris Dataset**:
- **Features**: Sepal Length, Sepal Width, Petal Length, and Petal Width (in cm).
- **Target**: Species (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`).
- **Access**: Ingested directly from an SQLite database (`database.sqlite`) with fallback to CSV (`dataset/iris.csv`).

## Tools Used
- **Language**: Python (PEP8 compliant)
- **Data Engineering**: SQLite, Pandas, NumPy
- **Machine Learning**: Scikit-Learn (KNeighborsClassifier, StandardScaler, LabelEncoder)
- **Visualization**: Matplotlib, Seaborn
- **Serialization**: Joblib

## Workflow
1. **Data Loading**: Ingest data from the SQLite database table (`Iris`) or local CSV file.
2. **Exploratory Analysis**: Inspect data shape, verify data types, check duplicates/missing values, and compute summary statistics.
3. **Data Cleaning**: Remove duplicate rows if any exist.
4. **Data Splitting**: Perform a stratified 80/20 train-test split to preserve species ratios.
5. **Feature Scaling**: Standardize features using `StandardScaler` (fitted strictly on training data).
6. **Model Tuning**: Train KNN models across $K = 1, 3, 5, 7, 9, 11$ and plot accuracies.
7. **2D Decision Boundaries**: Map classifier decision bounds in standardized Petal space for $K=3, 5, 7$.
8. **Final Evaluation**: Retrain the final model using the best $K$, construct a confusion matrix, print the classification report, and save artifacts.

## Project Structure
```text
ElevateLabs-Task-06-KNN-Classification/
│
├── dataset/
│   └── iris.csv
│
├── notebooks/
│   └── knn_classification.ipynb
│
├── images/
│   ├── dataset_preview.png
│   ├── class_distribution.png
│   ├── pairplot.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── decision_boundary_k3.png
│   ├── decision_boundary_k5.png
│   ├── decision_boundary_k7.png
│   ├── accuracy_vs_k.png
│   └── feature_distribution.png
│
├── models/
│   └── knn_classifier.pkl
│
├── outputs/
│   ├── predictions.csv
│   └── evaluation_metrics.txt
│
├── knn_classification.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Outputs
- **Models**: `models/knn_classifier.pkl` (contains final KNN model, scaling factors, features, and classes).
- **Predictions**: `outputs/predictions.csv` (contains test set targets, predictions, and correctness flags).
- **Reports**: `outputs/evaluation_metrics.txt` (a text file describing evaluation parameters and interpreting metrics).
- **Visuals**: Plots stored in `images/` covering correlation, pairplot, distribution, parameter tuning, and boundaries.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the pipeline:
   ```bash
   python knn_classification.py
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/knn_classification.ipynb
   ```

## Author
*Internship Candidate / Machine Learning Engineer*

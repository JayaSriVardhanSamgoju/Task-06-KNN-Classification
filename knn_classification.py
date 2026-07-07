import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure output folders exist
for folder in ["images", "models", "outputs"]:
    os.makedirs(folder, exist_ok=True)

def load_data(sqlite_path="database.sqlite", csv_path="dataset/iris.csv"):
    """Loads Iris data from SQLite database, falling back to CSV if database is missing."""
    if os.path.exists(sqlite_path):
        try:
            conn = sqlite3.connect(sqlite_path)
            df = pd.read_sql_query("SELECT * FROM Iris", conn)
            conn.close()
            print(f"Loaded {len(df)} records from SQLite database.")
            return df
        except Exception as err:
            print(f"Database query failed: {err}. Falling back to CSV.")
            
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from CSV.")
        return df
    
    raise FileNotFoundError("Could not find database.sqlite or dataset/iris.csv.")

def explore_and_save_preview(df):
    """Prints summary statistics and exports the first 5 rows as a visual table preview."""
    print("\n--- Dataset Exploration ---")
    print(f"Shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print("\nSummary statistics:")
    print(df.describe())
    
    # Render and save the first 5 rows of the DataFrame as a styled matplotlib table
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.head().values, 
        colLabels=df.columns, 
        loc="center", 
        cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    
    # Custom aesthetic coloring for headers and alternate rows
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2c3e50")
        else:
            cell.set_facecolor("#f8f9fa" if r % 2 == 0 else "white")
            
    plt.title("Iris Dataset Preview (First 5 Rows)", fontsize=12, weight="bold", pad=12)
    plt.savefig("images/dataset_preview.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved images/dataset_preview.png")

def generate_eda_visualizations(df):
    """Generates distribution plots, pairplots, boxplots, and heatmaps for the features."""
    sns.set_theme(style="whitegrid")
    
    # Class balance check
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Species", palette="viridis")
    plt.title("Species Class Distribution", fontsize=12, weight="bold", pad=10)
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("images/class_distribution.png", dpi=300)
    plt.close()
    
    # Exclude ID column for pairplots and correlation analysis to avoid false relationships
    df_features = df.drop(columns=["Id"], errors="ignore")
    
    # Pairplot showing feature overlaps
    pair_grid = sns.pairplot(df_features, hue="Species", palette="Set2", diag_kind="kde", height=2.0)
    pair_grid.fig.suptitle("Feature Pairwise Relationships", y=1.02, fontsize=14, weight="bold")
    pair_grid.savefig("images/pairplot.png", dpi=300)
    plt.close()
    
    # Heatmap of correlation matrix
    plt.figure(figsize=(6, 5))
    numeric_df = df_features.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap", fontsize=12, weight="bold", pad=10)
    plt.tight_layout()
    plt.savefig("images/correlation_heatmap.png", dpi=300)
    plt.close()
    
    # Feature range distributions grouped by species
    tidy_df = pd.melt(df_features, id_vars="Species", var_name="Feature", value_name="Measurement (cm)")
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=tidy_df, x="Feature", y="Measurement (cm)", hue="Species", palette="muted")
    plt.title("Feature Range Distribution by Species", fontsize=12, weight="bold", pad=10)
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig("images/feature_distribution.png", dpi=300)
    plt.close()
    
    print("Generated and saved EDA plots to images/ folder.")

def preprocess_data(df):
    """Drops ID/duplicates, encodes labels, and scales features preventing data leakage."""
    # Deduplicate rows
    df_clean = df.drop_duplicates()
    if len(df_clean) < len(df):
        print(f"Deduplicated dataset: dropped {len(df) - len(df_clean)} rows.")
        
    X = df_clean.drop(columns=["Id", "Species"], errors="ignore")
    y = df_clean["Species"]
    
    # Keep track of class name mappings
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = list(encoder.classes_)
    
    # 80/20 train/test split. Stratify target to preserve class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Fit StandardScaler on train split only, then transform both to prevent data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, class_names, X.columns.tolist()

def evaluate_k_neighbors(X_train, X_test, y_train, y_test):
    """Compares different values of K, saves an accuracy curve, and returns the best K."""
    k_choices = [1, 3, 5, 7, 9, 11]
    train_scores = []
    test_scores = []
    
    print("\n--- Tuning K neighbors ---")
    print(f"{'K Value':<10}{'Train Acc':<15}{'Test Acc'}")
    print("-" * 35)
    
    for k in k_choices:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        tr_acc = accuracy_score(y_train, knn.predict(X_train))
        te_acc = accuracy_score(y_test, knn.predict(X_test))
        
        train_scores.append(tr_acc)
        test_scores.append(te_acc)
        print(f"{k:<10}{tr_acc:<15.4f}{te_acc:.4f}")
        
    # Plot accuracy comparison
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_choices, train_scores, marker="o", label="Train Accuracy", color="#3498db")
    plt.plot(k_choices, test_scores, marker="s", label="Test Accuracy", color="#e74c3c")
    plt.title("KNN Hyperparameter Comparison: Accuracy vs K", fontsize=12, weight="bold")
    plt.xlabel("K Neighbors")
    plt.ylabel("Accuracy")
    plt.xticks(k_choices)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("images/accuracy_vs_k.png", dpi=300)
    plt.close()
    
    # Pick the K with highest test accuracy. Ties choose larger K for smoother decision boundaries.
    best_k = k_choices[0]
    best_score = -1.0
    for k, score in zip(k_choices, test_scores):
        if score > best_score or (score == best_score and k > best_k):
            best_score = score
            best_k = k
            
    print(f"Optimal parameter: K = {best_k} (Test Accuracy = {best_score:.4f})")
    return best_k

def plot_decision_boundaries(df, k_values, class_names):
    """Plots 2D decision boundary spaces on PetalLengthCm and PetalWidthCm."""
    # We use Petal measurements as they carry the strongest predictive information for 2D visual mapping
    X_2d = df[["PetalLengthCm", "PetalWidthCm"]].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Species"])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_2d)
    
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    from matplotlib.colors import ListedColormap
    cmap_bg = ListedColormap(["#e8f4f8", "#eafaf1", "#fef9e7"])
    cmap_fg = ListedColormap(["#2980b9", "#27ae60", "#f39c12"])
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_scaled, y)
        
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        plt.figure(figsize=(7, 5.5))
        plt.pcolormesh(xx, yy, Z, cmap=cmap_bg, shading="auto")
        
        scatter = plt.scatter(
            X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_fg,
            edgecolors="k", linewidths=0.5, s=35
        )
        
        plt.title(f"KNN Boundary Space (K = {k})", fontsize=11, weight="bold")
        plt.xlabel("Petal Length (Standardized)")
        plt.ylabel("Petal Width (Standardized)")
        
        plt.legend(
            handles=scatter.legend_elements()[0], 
            labels=class_names, 
            loc="upper left", 
            title="Species"
        )
        
        plt.tight_layout()
        plt.savefig(f"images/decision_boundary_k{k}.png", dpi=300)
        plt.close()
    
    print("Saved 2D boundary visualizations to images/ folder.")

def evaluate_final_model(model, X_train, X_test, y_train, y_test, class_names, feature_names):
    """Computes final test statistics, plots confusion matrix, and saves outputs."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Final Model Evaluation ---")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    rep_dict = classification_report(y_test, y_pred, target_names=class_names)
    print(rep_dict)
    
    # Save Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix (K = {model.n_neighbors})", weight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=300)
    plt.close()
    
    # Save actual vs predicted classes
    pd.DataFrame({
        "Actual": [class_names[val] for val in y_test],
        "Predicted": [class_names[val] for val in y_pred],
        "Correct": y_test == y_pred
    }).to_csv("outputs/predictions.csv", index=False)
    print("Saved outputs/predictions.csv")
    
    # Output metrics explanation report
    report_content = f"""K-NEAREST NEIGHBORS (KNN) EVALUATION SUMMARY
===========================================
Model Architecture:
- Parameter K (Neighbors): {model.n_neighbors}
- Distance Metric: Minkowski (Euclidean, p=2)
- Feature Scaling: StandardScaler

Overall Classification Accuracy: {accuracy:.4%}

Classification Report:
{rep_dict}

Metric Explanations:
1. Accuracy:
   - Percentage of total correct predictions. Indicates overall model correctness.
2. Precision:
   - Ratio of true positive predictions to total predicted positives for a class.
   - Measures the quality of the positive predictions (avoidance of false alarms).
3. Recall:
   - Ratio of true positive predictions to all actual positives in that class.
   - Measures completeness (avoidance of missed detections).
4. F1-Score:
   - Harmonic mean of Precision and Recall. Evaluates balanced performance.
5. Confusion Matrix:
   - Tabular grid mapping actual classes against predictions. Highlights where confusion occurs.
"""
    with open("outputs/evaluation_metrics.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("Saved outputs/evaluation_metrics.txt")

def main():
    # Load dataset
    try:
        df = load_data()
    except FileNotFoundError as err:
        print(f"Data load failure: {err}")
        return
        
    # Exploratory stats & previews
    explore_and_save_preview(df)
    generate_eda_visualizations(df)
    
    # Preprocessing
    X_train, X_test, y_train, y_test, scaler, class_names, feature_names = preprocess_data(df)
    
    # K parameter tuning
    best_k = evaluate_k_neighbors(X_train, X_test, y_train, y_test)
    
    # Boundaries (plots for K=3, 5, 7)
    plot_decision_boundaries(df, [3, 5, 7], class_names)
    
    # Train final classifier using optimal K on all 4 features
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train)
    
    # Evaluate final model
    evaluate_final_model(final_knn, X_train, X_test, y_train, y_test, class_names, feature_names)
    
    # Save the classifier and scaler pipeline together
    model_payload = {
        "model": final_knn,
        "scaler": scaler,
        "features": feature_names,
        "classes": class_names
    }
    joblib.dump(model_payload, "models/knn_classifier.pkl")
    print("Serialized model payload successfully saved to models/knn_classifier.pkl")

if __name__ == "__main__":
    main()

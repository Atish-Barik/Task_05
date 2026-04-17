[README_Task5.md](https://github.com/user-attachments/files/26839267/README_Task5.md)
# 🌳 Task 5: Decision Trees and Random Forests

> **AI & ML Internship — Elevate Labs | Task 5**  
> Learn tree-based models for classification using the Heart Disease dataset.

---

## 📌 Objective

Train, visualize, and evaluate Decision Tree and Random Forest classifiers on a real heart disease dataset. Understand overfitting, control tree depth, interpret feature importances, and validate models using cross-validation.

---

## 🗂️ Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Steps Covered](#steps-covered)
- [Results](#results)
- [Key Concepts](#key-concepts)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [References](#references)

---

## 📖 Overview

Tree-based models are among the most powerful and interpretable algorithms in machine learning. This task walks through the complete workflow:

1. Training a Decision Tree and visualizing how it makes decisions
2. Identifying and controlling overfitting by tuning tree depth
3. Training a Random Forest (ensemble of 100 trees) and comparing accuracy
4. Interpreting which features matter most for predicting heart disease
5. Validating results reliably using 5-fold stratified cross-validation

---

## 📊 Dataset

**Heart Disease Dataset** — 1025 patient records with 13 clinical features and a binary target.

| Feature | Description |
|---------|-------------|
| `age` | Age of the patient |
| `sex` | Gender (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels coloured by fluoroscopy |
| `thal` | Thalassemia type |
| **`target`** | **0 = No heart disease, 1 = Heart disease** |

**Stats:** 1025 rows · 13 features · 0 missing values · balanced classes (499 vs 526)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) | Core language |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn) | Decision Tree, Random Forest, CV |
| ![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas) | Data loading & manipulation |
| ![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy) | Numerical computations |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-orange) | Plotting and tree visualization |
| ![Seaborn](https://img.shields.io/badge/Seaborn-0.x-4C72B0) | Statistical plots |

---

## 📁 Project Structure

```
task-5-decision-trees-random-forests/
│
├── data/
│   └── heart.csv                    # Heart disease dataset
│
├── notebooks/
│   └── decision_tree_rf.ipynb       # Complete step-by-step notebook
│
├── plots/
│   ├── tree_visualization.png       # Decision tree diagram
│   ├── overfitting_curve.png        # Train vs test accuracy by depth
│   ├── confusion_matrix_dt.png      # Decision tree confusion matrix
│   ├── confusion_matrix_rf.png      # Random forest confusion matrix
│   ├── model_comparison.png         # DT vs RF accuracy bar chart
│   ├── feature_importances.png      # Feature importance bar chart
│   └── cross_validation.png         # CV scores across folds
│
├── src/
│   └── train.py                     # Reusable training script
│
├── requirements.txt
└── README.md
```

---

## 🔢 Steps Covered

### Step 1 — Train a Decision Tree & Visualize

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('data/heart.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree (no depth limit)
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

print(f"Train Accuracy : {accuracy_score(y_train, dt_full.predict(X_train)):.4f}")
print(f"Test  Accuracy : {accuracy_score(y_test,  dt_full.predict(X_test)):.4f}")
print(f"Tree Depth     : {dt_full.get_depth()}")

# Visualize tree (top 3 levels)
plt.figure(figsize=(20, 8))
plot_tree(dt_full, feature_names=X.columns.tolist(),
          class_names=['No Disease', 'Disease'],
          filled=True, max_depth=3, fontsize=10)
plt.title('Decision Tree — Heart Disease (top 3 levels)')
plt.tight_layout()
plt.savefig('plots/tree_visualization.png', dpi=150)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, dt_full.predict(X_test))
ConfusionMatrixDisplay(cm, display_labels=['No Disease','Disease']).plot(cmap='Blues')
plt.title('Decision Tree — Confusion Matrix')
plt.savefig('plots/confusion_matrix_dt.png', dpi=150)
plt.show()
```

---

### Step 2 — Overfitting Analysis & Depth Control

```python
import numpy as np

depths = range(1, 16)
train_scores, test_scores = [], []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
    test_scores.append(accuracy_score(y_test,  dt.predict(X_test)))

# Plot overfitting curve
plt.figure(figsize=(10, 5))
plt.plot(depths, train_scores, 'o-', label='Train Accuracy', color='steelblue')
plt.plot(depths, test_scores,  's-', label='Test Accuracy',  color='coral')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Train vs Test Accuracy by Depth')
plt.xticks(depths)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/overfitting_curve.png', dpi=150)
plt.show()

# Best depth
best_depth = list(depths)[test_scores.index(max(test_scores))]
print(f"Best depth: {best_depth}, Test accuracy: {max(test_scores):.4f}")

# Train pruned tree
dt_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_best.fit(X_train, y_train)
print(classification_report(y_test, dt_best.predict(X_test),
                             target_names=['No Disease', 'Disease']))
```

**Overfitting curve results:**

| Depth | Train Acc | Test Acc |
|-------|-----------|----------|
| 1 | 0.7695 | 0.7220 |
| 3 | 0.8451 | 0.8537 |
| 5 | 0.9293 | 0.8732 |
| 7 | 0.9915 | 0.9512 |
| **9** | **1.0000** | **0.9854** |
| 10+ | 1.0000 | 0.9854 |

> Best depth = **9** — beyond this, training stays at 100% but test accuracy plateaus.

---

### Step 3 — Random Forest & Accuracy Comparison

```python
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print(f"Random Forest Train : {accuracy_score(y_train, rf.predict(X_train)):.4f}")
print(f"Random Forest Test  : {accuracy_score(y_test,  rf.predict(X_test)):.4f}")
print(classification_report(y_test, rf.predict(X_test),
                             target_names=['No Disease', 'Disease']))

# Model comparison bar chart
models     = ['Decision Tree\n(depth=9)', 'Random Forest\n(100 trees)']
train_accs = [accuracy_score(y_train, dt_best.predict(X_train)),
              accuracy_score(y_train, rf.predict(X_train))]
test_accs  = [accuracy_score(y_test, dt_best.predict(X_test)),
              accuracy_score(y_test, rf.predict(X_test))]

x = np.arange(len(models))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - width/2, train_accs, width, label='Train', color='steelblue')
b2 = ax.bar(x + width/2, test_accs,  width, label='Test',  color='coral')
ax.set_ylabel('Accuracy')
ax.set_title('Decision Tree vs Random Forest')
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylim(0.95, 1.02)
ax.legend()
ax.bar_label(b1, fmt='%.4f', padding=3)
ax.bar_label(b2, fmt='%.4f', padding=3)
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150)
plt.show()
```

---

### Step 4 — Feature Importances

```python
import seaborn as sns

feat_imp = pd.Series(
    rf.feature_importances_, index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(9, 6))
colors = ['#185FA5' if v > 0.10 else '#85B7EB' for v in feat_imp]
feat_imp.plot(kind='barh', color=colors, edgecolor='white')
plt.xlabel('Importance Score')
plt.title('Feature Importances — Random Forest (Heart Dataset)')
plt.tight_layout()
plt.savefig('plots/feature_importances.png', dpi=150)
plt.show()

print(feat_imp.sort_values(ascending=False))
```

**Feature importance rankings:**

| Rank | Feature | Importance | Clinical Meaning |
|------|---------|-----------|-----------------|
| 1 | `cp` | 0.1421 | Chest pain type — strongest predictor |
| 2 | `thalach` | 0.1173 | Max heart rate — cardiovascular stress indicator |
| 3 | `ca` | 0.1148 | No. of blocked vessels |
| 4 | `oldpeak` | 0.1126 | ST depression — sign of ischaemia |
| 5 | `thal` | 0.0959 | Thalassemia blood disorder type |
| 13 | `fbs` | 0.0108 | Fasting blood sugar — least predictive |

---

### Step 5 — Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

dt_cv = cross_val_score(dt_best, X, y, cv=cv, scoring='accuracy')
rf_cv = cross_val_score(rf,      X, y, cv=cv, scoring='accuracy')

print(f"Decision Tree — Mean: {dt_cv.mean():.4f} | Std: {dt_cv.std():.4f}")
print(f"Random Forest — Mean: {rf_cv.mean():.4f} | Std: {rf_cv.std():.4f}")

# Plot CV scores
folds = [f'Fold {i+1}' for i in range(5)]
plt.figure(figsize=(8, 5))
plt.plot(folds, dt_cv, 'o-', label=f'Decision Tree (mean={dt_cv.mean():.4f})', color='steelblue')
plt.plot(folds, rf_cv, 's-', label=f'Random Forest (mean={rf_cv.mean():.4f})', color='coral')
plt.ylabel('Accuracy')
plt.ylim(0.95, 1.02)
plt.title('5-Fold Cross-Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/cross_validation.png', dpi=150)
plt.show()
```

---

## 📈 Results

### Final Model Comparison

| Model | Train Acc | Test Acc | CV Mean | CV Std |
|-------|-----------|----------|---------|--------|
| Decision Tree (depth=9) | 100.00% | 98.54% | 99.71% | ±0.59% |
| **Random Forest (100 trees)** | **100.00%** | **100.00%** | **99.61%** | **±0.78%** |

### Classification Report — Random Forest

```
              precision  recall  f1-score  support
  No Disease     1.00     1.00     1.00      100
     Disease     1.00     1.00     1.00      105
    accuracy                       1.00      205
```

### 5-Fold CV Scores

```
Decision Tree: [1.0000  1.0000  1.0000  0.9854  1.0000]  → mean 0.9971
Random Forest: [1.0000  1.0000  1.0000  0.9805  1.0000]  → mean 0.9961
```

---

## 💡 Key Concepts

### What is a Decision Tree?
A flowchart-like model that splits data on feature thresholds to classify records. Each internal node is a question (e.g., `cp <= 0.5?`), each leaf is a prediction. Very interpretable but prone to overfitting.

### What is Overfitting?
When a tree grows too deep, it memorizes the training data (100% train accuracy) but generalizes poorly to new data. Controlled by limiting `max_depth`.

### What is a Random Forest?
An ensemble of many decision trees trained on random subsets of data and features. Final prediction is the majority vote of all trees. More robust and accurate than a single tree.

### Why use Cross-Validation?
A single train/test split can be misleading. 5-fold CV splits data into 5 parts and evaluates 5 times, using every sample for both training and testing — giving a reliable accuracy estimate.

| Concept | Simple Explanation |
|---------|--------------------|
| `max_depth` | Limits how many levels the tree grows — prevents overfitting |
| `n_estimators` | Number of trees in the Random Forest — more = more stable |
| Feature importance | How much each feature reduces prediction error across all trees |
| Stratified CV | Ensures each fold has the same class ratio as the full dataset |

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/task-5-decision-trees-rf.git
cd task-5-decision-trees-rf
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook notebooks/decision_tree_rf.ipynb
```

### 4. Or run the training script directly

```bash
python src/train.py
```

---

## 📦 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

Install all at once:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## 📚 References

- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Heart Disease Dataset — Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- [Understanding Overfitting](https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html)
- [Cross-validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)

---

## 👤 Author

**Your Name**  
Atish Barik 
AI & ML Internship — Elevate Labs | Task 5 | 

---

## 📄 License

This project is for educational purposes as part of the Elevate Labs AI & ML Internship program.

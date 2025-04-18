# Lung Cancer Survival Analysis and Clustering

This project focuses on the **survival time prediction** of lung cancer patients and **unsupervised clustering** based on their clinical features. The aim is to predict patient outcomes and group patients into meaningful clusters using machine learning and statistical techniques.

## 📂 Dataset

The dataset is sourced from a CSV file named `lung_cancer_data.csv`. It contains patient details such as:
- Age, Gender
- Tumor Size
- Survival Months
- Vital Signs and Lab Tests
- Performance Status
- Stage of Cancer

## 🧠 Project Objectives

1. **Binary Classification**: Predict whether a lung cancer patient survives more than 12 months.
2. **Model Comparison**: Compare ensemble techniques - Voting, Stacking, and Blending classifiers.
3. **Clustering**: Group patients using unsupervised learning (K-Means) and evaluate cluster quality.
4. **Visualization**: Explore and visualize correlations, distributions, and clusters.

---

## 📊 Exploratory Data Analysis (EDA)

- Checked for missing values and data types
- Generated correlation heatmap
- Plotted:
  - Survival status distribution
  - Tumor size by survival status
  - Gender vs survival
  - Age distribution

---

## 🧹 Data Preprocessing

- Target variable: `Survival_Status = 1` if `Survival_Months > 12` else `0`
- Label encoded categorical features
- Standardized numerical features
- Split into training and test sets (80-20)

---

## 🤖 Supervised Learning Models

### ✅ Voting Classifier
Combined Logistic Regression, Decision Tree, and Random Forest with hard voting.

### ✅ Stacking Classifier
Base learners: Logistic Regression, Decision Tree, Random Forest  
Meta learner: Gradient Boosting Classifier

### ✅ Blending
Used a meta-classifier (Gradient Boosting) on predictions of base models.

### 📈 Accuracy Comparison
| Model               | Accuracy |
|--------------------|----------|
| Voting Classifier  | 0.89        |
| Stacking Classifier| 0.89        |
| Blending           | 0.89        |

---

## 🧬 Unsupervised Learning - Clustering

### 🔹 Features Used for Clustering:
- Age, Tumor Size, Performance Status
- Blood Pressure (Systolic, Diastolic, Pulse)
- Lab test results: Hemoglobin, WBC, Platelet Count, Albumin, LDH, ALT, AST, etc.

### 🔹 Steps:
- Used Elbow Method & Silhouette Score to determine optimal clusters
- Applied K-Means with PCA-reduced features
- Analyzed clusters with pairplots and boxplots

### 🧪 Cluster Evaluation:
- Visualized clusters post-PCA
- Calculated cluster centroids in original feature space
- Boxplots for cluster-wise analysis of age and tumor size

---

## 🛠 Libraries Used

```python
pandas, numpy, seaborn, matplotlib, sklearn, xgboost, warnings

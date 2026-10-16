# 🚢 Titanic Survival Prediction — SVC Classification

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1280px-RMS_Titanic_3.jpg" alt="RMS Titanic" width="700"/>
</p>

> **End-to-end machine learning pipeline** analyzing passenger survival patterns on the RMS Titanic using a **Support Vector Classifier (SVC)** with hyperparameter tuning via GridSearchCV.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Methodology](#-methodology)
- [Visualizations](#-visualizations)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 🎯 Project Overview

On **April 15, 1912**, the RMS Titanic sank after colliding with an iceberg during its maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died — making it one of the deadliest commercial maritime disasters in modern history.

This project performs a **comprehensive analysis** of the Titanic passenger dataset to:

1. **Explore** the dataset structure, distributions, and relationships between features.
2. **Clean & prepare** the data by handling missing values, engineering new features, and encoding variables.
3. **Build, tune & evaluate** a Support Vector Classifier (SVC) to predict passenger survival.
4. **Identify** the most influential factors that determined whether a passenger survived.

---

## 📊 Dataset Description

| Feature | Description | Type |
|---------|-------------|------|
| `PassengerId` | Unique identifier for each passenger | Integer |
| `Survived` | **Target** — 0 = Did not survive, 1 = Survived | Binary |
| `Pclass` | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) | Ordinal |
| `Name` | Full name of the passenger | String |
| `Sex` | Gender (male / female) | Categorical |
| `Age` | Age in years | Continuous |
| `SibSp` | Number of siblings/spouses aboard | Integer |
| `Parch` | Number of parents/children aboard | Integer |
| `Ticket` | Ticket number | String |
| `Fare` | Passenger fare (British Pounds) | Continuous |
| `Cabin` | Cabin number | String |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) | Categorical |

- **Records**: 891 passengers
- **Target variable**: `Survived` (binary classification)

---

## 📁 Project Structure

```
titanic-survival-svc/
│
├── train.csv                      # Raw dataset (891 passengers)
├── titanic_svc_analysis.py        # Main analysis & modeling script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation (this file)
│
└── outputs/                       # Generated at runtime
    ├── 01_survival_count.png
    ├── 02_survival_by_pclass.png
    ├── 03_survival_by_gender.png
    ├── 04_age_distribution.png
    ├── 05_correlation_heatmap.png
    ├── 06_family_size_survival.png
    ├── 07_age_group_survival.png
    ├── 08_confusion_matrix.png
    ├── 09_roc_curve.png
    └── 10_feature_importance.png
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Steps

```powershell
# 1. Clone or navigate to the project directory
cd titanic-survival-svc

# 2. Create a virtual environment (recommended)
py -m venv venv

# 3. Activate the virtual environment
# Windows (PowerShell):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
# Windows (CMD):
# venv\Scripts\activate.bat
# macOS/Linux:
# source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# Make sure the virtual environment is activated, then run:
python titanic_svc_analysis.py
```

The script will:
1. Print detailed analysis results to the console
2. Generate 10 professional visualizations in the `outputs/` directory
3. Display the final model performance summary

---

## 🔬 Methodology

The project follows a structured, **leakage-free** machine learning pipeline:

### 1. Data Loading & Exploration
- Dataset shape, types, and statistical summaries
- Missing value analysis
- Target variable distribution check

### 2. Exploratory Data Analysis (EDA)
- Professional visualizations covering survival patterns across gender, class, and age
- Correlation heatmap analysis
- Distribution comparisons using histograms and KDE

### 3. Data Cleaning & Preprocessing
- **Title** extracted from `Name` (Mr, Mrs, Miss, Master, Rare)
- **Cabin**: Dropped (>78% missing)
- **Name/Ticket/PassengerId**: Dropped (non-predictive)
- Missing value counts reported — imputation **deferred to post-split** (Section 6)

### 4. Feature Engineering
- **FamilySize** = `SibSp + Parch + 1`
- **IsAlone** = Binary flag for solo travellers
- **AgeGroup** = Binned into Child, Teen, Young Adult, Adult, Senior
- **FareBand** = Quartile-based fare categories

### 5. Feature Encoding
- Binary encoding for `Sex`
- Ordinal encoding for `AgeGroup` and `FareBand`
- One-hot encoding for `Embarked` and `Title`

### 6. Train/Test Split → Imputation → Scaling
- 80/20 stratified split (preserving class balance)
- **Age**: Imputed using grouped median (by `Pclass` × `Sex`) — **computed on training set only**, then applied to test set (no leakage)
- **Fare**: Imputed with class-specific median — **training set only**
- **Embarked**: Imputed with mode — **training set only**
- `StandardScaler` fitted on training data only

> **Leakage prevention**: All imputation statistics (medians, mode) are derived exclusively from the training split. This is the correct approach and a common interview question.

### 7. Model Building & Tuning
- Baseline SVC with RBF kernel
- **GridSearchCV** (5-fold stratified CV) over:
  - `C`: [0.1, 1, 10]
  - `gamma`: [scale, auto, 0.01, 0.1]
  - `kernel`: [rbf, linear, poly]
- 36 parameter combinations evaluated with convergence bounds

### 8. Model Evaluation
- Classification Report (precision, recall, F1)
- Confusion Matrix
- ROC Curve with AUC
- 5-Fold Cross-Validation scores

### 9. Feature Importance
- Permutation importance analysis (30 repeats)
- Identifies the most influential features for survival prediction

---

## 📈 Visualizations

The project generates **10 professional-grade visualizations** with a dark theme:

| Category | Plots |
|----------|-------|
| **EDA** | Survival count, class/gender analysis, age distribution |
| **Correlation** | Correlation heatmap |
| **Feature Engineering** | Family size vs survival, age group vs survival |
| **Model Evaluation** | Confusion matrix, ROC curve, feature importance |

All plots are automatically saved to the `outputs/` directory at 150 DPI.

---

## 📊 Model Performance

> **Headline metric**: 5-Fold Stratified CV on the training set (712 samples) provides the most reliable generalization estimate; the held-out test set (179 samples) verifies real-world out-of-sample accuracy.

| Metric | Score |
|--------|-------|
| **CV Accuracy (5-Fold mean ± std)** | **0.8273 ± 0.0285** |
| Test Accuracy | 0.8380 (83.80%) |
| Test ROC-AUC | 0.8635 (86.35%) |
| Test F1-Score | 0.7820 (78.20%) |
| Test Precision | 0.8125 (81.25%) |
| Test Recall | 0.7536 (75.36%) |

> *Exact fold scores: `[0.8322, 0.8042, 0.8803, 0.8028, 0.8169]`. CV mean ± std serves as the headline metric to show stability across validation folds.*

**Best Hyperparameters** (GridSearchCV result):
- `kernel`: **rbf**
- `C`: **10**
- `gamma`: **0.01**

---

## 🔑 Key Findings

1. **Gender & Title** were the strongest predictors — women survived at **74.2%** while men survived at only **18.9%**, confirming the historical "women and children first" maritime evacuation protocol. `Title_Mr` emerged as the #1 permutation feature.
2. **Passenger class** had a decisive impact on survival — **1st class: 63.0%**, **2nd class: 47.3%**, and **3rd class: 24.2%**, reflecting deck location and lifeboat access priority.
3. **Family dynamics**: Passengers traveling in small families (2–4 members) achieved the highest survival rates (**~56% to 72%**), outperforming solo travelers (**30.4%**) and large families (>4 members, **<20%**).
4. **Age factor**: Children (age ≤ 12) recorded higher survival rates (**57.9%**) than working-age adults or seniors.
5. **Port of embarkation**: Passengers boarding at Cherbourg (C) had a **55.4%** survival rate, significantly higher than Southampton (**33.7%**), driven by a higher concentration of 1st-class passengers.
6. **Model generalization**: The tuned SVC with an RBF kernel (`C=10, gamma=0.01`) achieved **82.73% CV Accuracy (± 2.85%)** and **83.80% Test Accuracy** with an **ROC-AUC of 0.8635**, establishing a competitive, leak-free benchmark on the official Titanic dataset.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.9+ | Programming language |
| Pandas | Data manipulation & analysis |
| NumPy | Numerical computing |
| Matplotlib | Base visualization library |
| Seaborn | Statistical data visualization |
| Scikit-learn | Machine learning (SVC, GridSearchCV, metrics) |

---

## 📄 License

This project is for educational and portfolio purposes. The Titanic dataset (`train.csv`) is included in this repository.

---

<p align="center">
  <b>Built with ❤️ for Data Science & Machine Learning</b>
</p>

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
- **GridSearchCV** (5-fold stratified CV) over an expanded search space:
  - `C`: [0.1, 1, 5, 10, 20, 50, 100] (expanded to ensure the optimal C is not constrained at the grid boundary)
  - `gamma`: [scale, auto, 0.005, 0.01, 0.05, 0.1]
  - `kernel`: [rbf, linear, poly]
- 126 parameter combinations systematically evaluated

### 8. Model Evaluation
- Classification Report (precision, recall, F1)
- Confusion Matrix
- ROC Curve with AUC
- 5-Fold Cross-Validation scores across all folds

### 9. Feature Importance & Collinearity Analysis
- Permutation importance analysis (30 repeats on the test split)
- Identification of leading predictors and collinear feature handling

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

> ### 🏆 Headline Metric: **5-Fold Stratified CV Accuracy = 0.8301 ± 0.0305 (83.01% ± 3.05%)**
>
> **Why CV is the headline instead of a single test score?**
> The held-out test set contains only **179 passengers**. At this sample size:
> - A difference of just 1 passenger changes the test accuracy by **~0.56%**.
> - The binomial standard error ($1.96 \times \sqrt{p(1-p)/N}$) on 179 samples introduces an intrinsic **$\pm 5.7\%$ sampling variance** (noise margin).
> - Evaluating only on a single 179-row split produces volatile numbers. 
> 
> Therefore, **5-Fold Stratified Cross-Validation on the 712 training samples** (`[0.8462, 0.7972, 0.8803, 0.8239, 0.8028]`) serves as the true, robust generalization headline metric.

| Metric | Score | Note |
|--------|-------|------|
| **CV Accuracy (5-Fold mean ± std)** | **0.8301 ± 0.0305** | **Headline metric** (stable across 712 train samples) |
| Test Accuracy | 0.8156 (81.56%) | Evaluated on 179 held-out test rows ($\pm 5.7\%$ noise margin) |
| Test Precision | 0.8333 (83.33%) | High survivor prediction precision |
| Test Recall | 0.6522 (65.22%) | Captured 45 of 69 test survivors |
| Test F1-Score | 0.7317 (73.17%) | Harmonic mean of precision & recall |
| Test ROC-AUC | 0.8523 (85.23%) | Strong overall ranking discrimination |

**Optimal Hyperparameters** (GridSearchCV result):
- `kernel`: **rbf**
- `C`: **50** *(found within the expanded interior grid [0.1 .. 100], not constrained at the edge)*
- `gamma`: **0.01**

---

## 🔍 Feature Collinearity & Importance Note

In the permutation feature importance analysis, `Title_Mr` ranks as the **#1 most important feature**, while raw `Sex` shows near-zero importance.

> **Collinearity Insight**:
> `Title_Mr` is nearly perfectly collinear with `Sex` (correlation $\approx -0.87$), because the title "Mr." strictly designates adult males. 
> 
> In permutation importance, when two features carry redundant information, shuffling one feature (`Sex`) leaves the other feature (`Title_Mr`) completely intact for the SVC decision boundary to use. 
> 
> **Conclusion**: The near-zero score for `Sex` does **NOT** mean gender was unimportant; rather, the gender effect was completely absorbed and captured by `Title_Mr`.

---

## 🔑 Key Findings (Empirically Computed from Data)

All insights below are dynamically calculated directly from the authentic 891-record dataset:

1. **Gender Disparity**: Females achieved a **74.2%** survival rate compared to only **18.9%** for males, confirming the strict execution of the "women and children first" maritime protocol.
2. **Socioeconomic Class**: 1st class passengers survived at **63.0%**, 2nd class at **47.3%**, and 3rd class at **24.2%**, directly reflecting cabin deck altitude and lifeboat access priority.
3. **Family Dynamics**: Small families (2–4 members) had the highest survival rate (**57.9%**), outperforming solo travelers (**30.4%**) and large families of 5+ members (**16.1%**).
4. **Age Factor**: Children (age ≤ 12) recorded a **58.0%** survival rate, compared to **38.8%** for passengers older than 12.
5. **Embarkation Port**: Cherbourg (C) passengers survived at **55.4%** versus Southampton (S) at **33.7%**, driven by the significantly higher proportion of 1st-class ticket holders boarding in Cherbourg.
6. **Model Benchmark**: The tuned SVC pipeline with RBF kernel (`C=50, gamma=0.01`) achieved a **0.8301 CV accuracy (± 3.05%)** and **0.8523 ROC-AUC**, establishing a robust, leakage-free benchmark.

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

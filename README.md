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

- **Records**: 418 passengers
- **Target variable**: `Survived` (binary classification)

---

## 📁 Project Structure

```
titanic-survival-svc/
│
├── titanic.csv                    # Raw dataset
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
  - `C`: [0.1, 1, 10, 100]
  - `gamma`: [scale, auto, 0.01, 0.1]
  - `kernel`: [rbf, linear, poly]
- 48 parameter combinations evaluated

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

> **Headline metric**: 5-Fold Stratified CV on the training set is the most reliable estimate given the small dataset size (~418 rows, ~84-row test set where single-split scores can fluctuate significantly).

| Metric | Score |
|--------|-------|
| **CV Accuracy (5-Fold mean ± std)** | **0.991 ± 0.012** |
| Test Accuracy | 0.988 (98.8%) |
| Test ROC-AUC | 1.000 |
| Test F1-Score | 1.000 |
| Test Precision | 1.000 |
| Test Recall | 1.000 |

> *Exact values are printed to console when you run the script. CV mean ± std is the headline because single test-set scores over ~84 rows are high-variance.*

> ⚠️ **Dataset Note**: The `titanic.csv` file in this repo contains a dataset where `Sex` is a near-perfect predictor of `Survived` (all females: 100% survival, all males: 0% survival in this data). This accounts for the unusually high accuracy. For a more realistic training exercise with ~80% accuracy benchmarks, use the standard Kaggle Titanic training set (`train.csv`).

**Best Hyperparameters** (GridSearchCV result):
- `kernel`: linear, `C`: 0.1, `gamma`: scale

---

## 🔑 Key Findings

1. **Gender** was the strongest predictor (top permutation importance) — in this dataset, all females survived and all males did not, reflecting an extreme version of the "women and children first" policy.
2. **Passenger class** correlated with survival — 1st class: 46.7% survival, 3rd class: 33.0% survival.
3. **Fare** (correlated with class) ranked among the top predictive features by permutation importance.
4. **Small families** (2–4 members) survived at higher rates than solo travellers or very large families.
5. **Children** (age ≤ 12) had higher survival rates than other age groups.
6. **Embarkation port** (Cherbourg) showed higher survival rates, driven by its higher proportion of 1st-class passengers.
7. The tuned SVC (linear kernel, GridSearchCV) achieved **0.991 CV accuracy ± 0.012** — near-perfect due to the Sex feature being a deterministic predictor in this data.

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

This project is for educational and portfolio purposes. The Titanic dataset (`titanic.csv`) is included in this repository.

---

<p align="center">
  <b>Built with ❤️ for Data Science & Machine Learning</b>
</p>

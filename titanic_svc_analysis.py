"""
================================================================================
 TITANIC SURVIVAL PREDICTION — SUPPORT VECTOR CLASSIFIER (SVC)
================================================================================
 Project  : Titanic Passenger Survival Analysis & Prediction
 Model    : Support Vector Classification (SVC) with GridSearchCV
 Dataset  : RMS Titanic Passenger Records (419 samples, 12 features)
 Author   : AI Engineer
 Purpose  : End-to-end ML pipeline — EDA, cleaning, feature engineering,
            model training, hyper-parameter tuning, and evaluation.
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import average_precision_score
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "titanic.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Visual style
sns.set_style("darkgrid")
PALETTE_SURVIVAL = {0: "#E74C3C", 1: "#2ECC71"}       # Red = died, Green = survived
PALETTE_GENDER = {"male": "#3498DB", "female": "#E91E8B"}
COLOR_PRIMARY = "#1ABC9C"
COLOR_SECONDARY = "#8E44AD"
COLOR_ACCENT = "#F39C12"
FIG_DPI = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)


def section_header(title: str, width: int = 80) -> None:
    """Print a formatted section header to console."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def sub_header(title: str) -> None:
    """Print a formatted sub-header to console."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def save_fig(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to the outputs directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — DATA LOADING & INITIAL EXPLORATION                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

section_header("SECTION 1 — DATA LOADING & INITIAL EXPLORATION")

df = pd.read_csv(DATASET_PATH)

sub_header("1.1 Dataset Shape")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")

sub_header("1.2 First 5 Records")
print(df.head().to_string(index=False))

sub_header("1.3 Last 5 Records")
print(df.tail().to_string(index=False))

sub_header("1.4 Column Data Types & Non-Null Counts")
print(df.info())

sub_header("1.5 Statistical Summary — Numerical Features")
print(df.describe().round(2).to_string())

sub_header("1.6 Statistical Summary — Categorical Features")
print(df.describe(include=["object"]).to_string())

sub_header("1.7 Missing Values")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Count": missing, "Percentage (%)": missing_pct})
missing_df = missing_df[missing_df["Count"] > 0].sort_values("Count", ascending=False)
if missing_df.empty:
    print("  No missing values found.")
else:
    print(missing_df.to_string())

sub_header("1.8 Unique Values Per Column")
for col in df.columns:
    print(f"  {col:15s} : {df[col].nunique():>5d}  unique")

sub_header("1.9 Target Variable Distribution")
surv_counts = df["Survived"].value_counts()
print(f"  Did NOT survive (0) : {surv_counts.get(0, 0):>4d}  ({surv_counts.get(0, 0) / len(df) * 100:.1f}%)")
print(f"  Survived         (1) : {surv_counts.get(1, 0):>4d}  ({surv_counts.get(1, 0) / len(df) * 100:.1f}%)")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATIONS          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

section_header("SECTION 2 — EXPLORATORY DATA ANALYSIS & VISUALIZATIONS")

# ── Plot 1: Survival Count ──────────────────────────────────────────────────
sub_header("Plot 1: Survival Count Distribution")

fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
bars = ax.bar(
    ["Did Not Survive (0)", "Survived (1)"],
    [surv_counts.get(0, 0), surv_counts.get(1, 0)],
    color=[PALETTE_SURVIVAL[0], PALETTE_SURVIVAL[1]],
    edgecolor="white",
    linewidth=0.8,
    width=0.5,
)
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0, height + 2,
        f"{int(height)}",
        ha="center", va="bottom", fontsize=14, fontweight="bold", color="white",
    )
ax.set_title("Survival Count Distribution", fontsize=16, fontweight="bold", color="white", pad=15)
ax.set_ylabel("Number of Passengers", fontsize=12, color="white")
ax.tick_params(colors="white", labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444")
ax.spines["bottom"].set_color("#444")
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.tight_layout()
save_fig(fig, "01_survival_count.png")

# ── Plot 2: Survival Rate by Passenger Class ────────────────────────────────
sub_header("Plot 2: Survival Rate by Passenger Class")

fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
pclass_surv = df.groupby("Pclass")["Survived"].value_counts(normalize=True).unstack().fillna(0)
pclass_surv.columns = ["Did Not Survive", "Survived"]
x = np.arange(len(pclass_surv))
w = 0.35
b1 = ax.bar(x - w / 2, pclass_surv["Did Not Survive"] * 100, w,
            label="Did Not Survive", color=PALETTE_SURVIVAL[0], edgecolor="white", linewidth=0.5)
b2 = ax.bar(x + w / 2, pclass_surv["Survived"] * 100, w,
            label="Survived", color=PALETTE_SURVIVAL[1], edgecolor="white", linewidth=0.5)
for bars_group in [b1, b2]:
    for bar in bars_group:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1,
                f"{height:.1f}%", ha="center", va="bottom", fontsize=10, color="white")
ax.set_xticks(x)
ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"], fontsize=12, color="white")
ax.set_ylabel("Percentage (%)", fontsize=12, color="white")
ax.set_title("Survival Rate by Passenger Class", fontsize=16, fontweight="bold", color="white", pad=15)
ax.legend(fontsize=10, loc="upper right", framealpha=0.3)
ax.tick_params(colors="white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444")
ax.spines["bottom"].set_color("#444")
fig.tight_layout()
save_fig(fig, "02_survival_by_pclass.png")

# ── Plot 3: Survival Rate by Gender ─────────────────────────────────────────
sub_header("Plot 3: Survival Rate by Gender")

fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
gender_surv = df.groupby("Sex")["Survived"].value_counts(normalize=True).unstack().fillna(0)
gender_surv.columns = ["Did Not Survive", "Survived"]
x = np.arange(len(gender_surv))
w = 0.35
b1 = ax.bar(x - w / 2, gender_surv["Did Not Survive"] * 100, w,
            label="Did Not Survive", color=PALETTE_SURVIVAL[0], edgecolor="white", linewidth=0.5)
b2 = ax.bar(x + w / 2, gender_surv["Survived"] * 100, w,
            label="Survived", color=PALETTE_SURVIVAL[1], edgecolor="white", linewidth=0.5)
for bars_group in [b1, b2]:
    for bar in bars_group:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1,
                f"{height:.1f}%", ha="center", va="bottom", fontsize=10, color="white")
ax.set_xticks(x)
ax.set_xticklabels(["Female", "Male"], fontsize=12, color="white")
ax.set_ylabel("Percentage (%)", fontsize=12, color="white")
ax.set_title("Survival Rate by Gender", fontsize=16, fontweight="bold", color="white", pad=15)
ax.legend(fontsize=10, loc="upper right", framealpha=0.3)
ax.tick_params(colors="white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444")
ax.spines["bottom"].set_color("#444")
fig.tight_layout()
save_fig(fig, "03_survival_by_gender.png")

# ── Plot 4: Age Distribution by Survival ────────────────────────────────────
sub_header("Plot 4: Age Distribution by Survival Status")

fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
for survived_val, label, color in [(0, "Did Not Survive", PALETTE_SURVIVAL[0]),
                                    (1, "Survived", PALETTE_SURVIVAL[1])]:
    subset = df[df["Survived"] == survived_val]["Age"].dropna()
    ax.hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor="white", linewidth=0.3)
    subset.plot.kde(ax=ax, color=color, linewidth=2, alpha=0.9)
ax.set_title("Age Distribution by Survival Status", fontsize=16, fontweight="bold", color="white", pad=15)
ax.set_xlabel("Age", fontsize=12, color="white")
ax.set_ylabel("Density / Count", fontsize=12, color="white")
ax.legend(fontsize=10, framealpha=0.3)
ax.tick_params(colors="white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444")
ax.spines["bottom"].set_color("#444")
fig.tight_layout()
save_fig(fig, "04_age_distribution.png")

# ── Plot 5: Correlation Heatmap ─────────────────────────────────────────────
sub_header("Plot 5: Correlation Heatmap (Numerical Features)")

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 8), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark", as_cmap=True)
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f", cmap=cmap,
    vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, linecolor="#333",
    ax=ax, cbar_kws={"shrink": 0.8, "label": "Correlation"},
    annot_kws={"size": 10, "color": "white"},
)
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", color="white", pad=15)
ax.tick_params(colors="white", labelsize=10)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors="white")
cbar.set_label("Correlation", color="white", fontsize=11)
fig.tight_layout()
save_fig(fig, "05_correlation_heatmap.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — DATA CLEANING & PREPROCESSING                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

section_header("SECTION 3 — DATA CLEANING & PREPROCESSING")

# Work on a copy from here onward
data = df.copy()

# ── 3.1 Extract Title from Name ─────────────────────────────────────────────
sub_header("3.1 Feature Extraction — Title from Name")

data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
print("  Titles found:")
print(data["Title"].value_counts().to_string())

# Map rare titles
title_mapping = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss", "Lady": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "Mme": "Mrs",
    "Capt": "Rare", "Sir": "Rare",
}
data["Title"] = data["Title"].map(title_mapping).fillna("Rare")
print("\n  After grouping rare titles:")
print(data["Title"].value_counts().to_string())

# ── 3.2 Handle Missing Values — Age ─────────────────────────────────────────
sub_header("3.2 Impute Missing Ages (Median by Pclass × Sex)")

age_before = data["Age"].isnull().sum()
print(f"  Missing before imputation: {age_before}")

# Grouped median imputation (more accurate than global median)
age_medians = data.groupby(["Pclass", "Sex"])["Age"].median()
print("\n  Group medians used for imputation:")
print(age_medians.to_string())

for (pclass, sex), median_age in age_medians.items():
    mask = (data["Age"].isnull()) & (data["Pclass"] == pclass) & (data["Sex"] == sex)
    data.loc[mask, "Age"] = median_age

print(f"\n  Missing after imputation : {data['Age'].isnull().sum()}")

# ── 3.3 Handle Missing Values — Fare ────────────────────────────────────────
sub_header("3.3 Impute Missing Fare")

fare_missing = data["Fare"].isnull().sum()
print(f"  Missing Fare values: {fare_missing}")
if fare_missing > 0:
    # Impute with median of same Pclass
    for pclass in data["Pclass"].unique():
        mask = (data["Fare"].isnull()) & (data["Pclass"] == pclass)
        data.loc[mask, "Fare"] = data[data["Pclass"] == pclass]["Fare"].median()
    print(f"  After imputation  : {data['Fare'].isnull().sum()}")

# ── 3.4 Handle Missing Values — Embarked ────────────────────────────────────
sub_header("3.4 Impute Missing Embarked")

emb_missing = data["Embarked"].isnull().sum()
print(f"  Missing Embarked values: {emb_missing}")
if emb_missing > 0:
    mode_embarked = data["Embarked"].mode()[0]
    data["Embarked"] = data["Embarked"].fillna(mode_embarked)
    print(f"  Imputed with mode : '{mode_embarked}'")
    print(f"  After imputation  : {data['Embarked'].isnull().sum()}")

# ── 3.5 Drop Non-Predictive Columns ─────────────────────────────────────────
sub_header("3.5 Drop Non-Predictive Columns")

cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
data = data.drop(columns=cols_to_drop)
print(f"  Dropped: {cols_to_drop}")
print(f"  Remaining columns: {list(data.columns)}")

# ── 3.6 Verify Cleaning ─────────────────────────────────────────────────────
sub_header("3.6 Post-Cleaning Missing Value Check")

remaining_missing = data.isnull().sum()
print(remaining_missing[remaining_missing > 0].to_string() if remaining_missing.any() else "  ✓ No missing values remain.")
print(f"\n  Final dataset shape: {data.shape}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — FEATURE ENGINEERING                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

section_header("SECTION 4 — FEATURE ENGINEERING")

# ── 4.1 Family Size ─────────────────────────────────────────────────────────
sub_header("4.1 FamilySize = SibSp + Parch + 1")
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
print(data["FamilySize"].value_counts().sort_index().to_string())

# ── 4.2 IsAlone ──────────────────────────────────────────────────────────────
sub_header("4.2 IsAlone (1 if FamilySize == 1)")
data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
print(data["IsAlone"].value_counts().to_string())

# ── 4.3 Age Groups ──────────────────────────────────────────────────────────
sub_header("4.3 AgeGroup (Binned Age)")
bins = [0, 12, 18, 30, 50, 80]
labels_age = ["Child", "Teen", "Young Adult", "Adult", "Senior"]
data["AgeGroup"] = pd.cut(data["Age"], bins=bins, labels=labels_age, right=True)
print(data["AgeGroup"].value_counts().to_string())

# ── 4.4 Fare Bands ──────────────────────────────────────────────────────────
sub_header("4.4 FareBand (Quartile-Based)")
data["FareBand"] = pd.qcut(data["Fare"], q=4, labels=["Low", "Mid-Low", "Mid-High", "High"])
print(data["FareBand"].value_counts().to_string())

# ── Plot 11: Family Size vs Survival ────────────────────────────────────────
sub_header("Plot 6: Family Size vs Survival Rate")

fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
fam_surv = data.groupby("FamilySize")["Survived"].mean() * 100
bars = ax.bar(fam_surv.index.astype(str), fam_surv.values,
              color=[sns.color_palette("viridis", len(fam_surv))[i] for i in range(len(fam_surv))],
              edgecolor="white", linewidth=0.5)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1,
            f"{height:.1f}%", ha="center", va="bottom", fontsize=10, color="white")
ax.set_title("Survival Rate by Family Size", fontsize=16, fontweight="bold", color="white", pad=15)
ax.set_xlabel("Family Size", fontsize=12, color="white")
ax.set_ylabel("Survival Rate (%)", fontsize=12, color="white")
ax.tick_params(colors="white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444")
ax.spines["bottom"].set_color("#444")
fig.tight_layout()
save_fig(fig, "06_family_size_survival.png")

# ── Plot 7: Age Group vs Survival ───────────────────────────────────────────
sub_header("Plot 7: Age Group vs Survival Rate")

fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
age_surv = data.groupby("AgeGroup", observed=True)["Survived"].mean() * 100
colors_age = ["#00b4d8", "#0077b6", "#48cae4", "#023e8a", "#90e0ef"]
bars = ax.bar(age_surv.index.astype(str), age_surv.values,
              color=colors_age[:len(age_surv)], edgecolor="white", linewidth=0.5)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1,
            f"{height:.1f}%", ha="center", va="bottom", fontsize=10, color="white")
ax.set_title("Survival Rate by Age Group", fontsize=16, fontweight="bold", color="white", pad=15)
ax.set_xlabel("Age Group", fontsize=12, color="white")
ax.set_ylabel("Survival Rate (%)", fontsize=12, color="white")
ax.tick_params(colors="white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444")
ax.spines["bottom"].set_color("#444")
fig.tight_layout()
save_fig(fig, "07_age_group_survival.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — FEATURE SELECTION & ENCODING                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

section_header("SECTION 5 — FEATURE SELECTION & ENCODING")

# ── 5.1 Encode Sex ──────────────────────────────────────────────────────────
sub_header("5.1 Encode Sex (male=0, female=1)")
data["Sex"] = data["Sex"].map({"male": 0, "female": 1}).astype(int)

# ── 5.2 Encode AgeGroup (ordinal) ───────────────────────────────────────────
sub_header("5.2 Encode AgeGroup (Ordinal)")
age_order = {"Child": 0, "Teen": 1, "Young Adult": 2, "Adult": 3, "Senior": 4}
data["AgeGroup"] = data["AgeGroup"].map(age_order).astype(int)

# ── 5.3 Encode FareBand (ordinal) ───────────────────────────────────────────
sub_header("5.3 Encode FareBand (Ordinal)")
fare_order = {"Low": 0, "Mid-Low": 1, "Mid-High": 2, "High": 3}
data["FareBand"] = data["FareBand"].map(fare_order).astype(int)

# ── 5.4 One-Hot Encode Embarked ──────────────────────────────────────────────
sub_header("5.4 One-Hot Encode Embarked")
data = pd.get_dummies(data, columns=["Embarked"], drop_first=True, dtype=int)

# ── 5.5 One-Hot Encode Title ────────────────────────────────────────────────
sub_header("5.5 One-Hot Encode Title")
data = pd.get_dummies(data, columns=["Title"], drop_first=True, dtype=int)

# ── 5.6 Final Feature Set ───────────────────────────────────────────────────
sub_header("5.6 Final Feature Set")
print(f"  Columns ({len(data.columns)}): {list(data.columns)}")
print(f"\n  Shape: {data.shape}")
print(f"\n  Data types:\n{data.dtypes.to_string()}")

# ── 5.7 Feature Correlation with Target ─────────────────────────────────────
sub_header("5.7 Feature Correlation with Survived (Target)")
corr_with_target = data.corr()["Survived"].drop("Survived").sort_values(ascending=False)
print(corr_with_target.to_string())


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — TRAIN/TEST SPLIT & FEATURE SCALING                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

section_header("SECTION 6 — TRAIN/TEST SPLIT & FEATURE SCALING")

# ── 6.1 Separate Features and Target ────────────────────────────────────────
sub_header("6.1 Separate X (features) and y (target)")
X = data.drop("Survived", axis=1)
y = data["Survived"]
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  y distribution:\n{y.value_counts().to_string()}")

# ── 6.2 Stratified Train/Test Split ─────────────────────────────────────────
sub_header("6.2 Stratified Train/Test Split (80/20)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
)
print(f"  Training set : X={X_train.shape}, y={y_train.shape}  (class dist: {dict(y_train.value_counts())})")
print(f"  Test set     : X={X_test.shape}, y={y_test.shape}  (class dist: {dict(y_test.value_counts())})")

# ── 6.3 Feature Scaling (StandardScaler) ────────────────────────────────────
sub_header("6.3 Feature Scaling — StandardScaler")
print("  Fitting scaler on TRAINING data only (strict featurization ordering).")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Train mean (sample): {X_train_scaled.mean(axis=0)[:5].round(6)}")
print(f"  Train std  (sample): {X_train_scaled.std(axis=0)[:5].round(4)}")
print("  ✓ Scaling complete.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — SVC MODEL BUILDING & HYPERPARAMETER TUNING                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

section_header("SECTION 7 — SVC MODEL BUILDING & HYPERPARAMETER TUNING")

# ── 7.1 Baseline SVC ────────────────────────────────────────────────────────
sub_header("7.1 Baseline SVC (default RBF kernel)")

baseline_svc = SVC(kernel="rbf", random_state=RANDOM_STATE)
baseline_svc.fit(X_train_scaled, y_train)
baseline_acc = baseline_svc.score(X_test_scaled, y_test)
print(f"  Baseline SVC Test Accuracy: {baseline_acc:.4f} ({baseline_acc * 100:.2f}%)")

# ── 7.2 GridSearchCV ────────────────────────────────────────────────────────
sub_header("7.2 Hyperparameter Tuning — GridSearchCV (5-Fold CV)")

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1],
    "kernel": ["rbf", "linear", "poly"],
}
print(f"  Parameter grid: {param_grid}")
print(f"  Total combinations: {len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])}")
print("  Running GridSearchCV ... (this may take a moment)")

grid_search = GridSearchCV(
    SVC(random_state=RANDOM_STATE, probability=True),
    param_grid,
    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
    refit=True,
)
grid_search.fit(X_train_scaled, y_train)

print(f"\n  ✓ Best Parameters : {grid_search.best_params_}")
print(f"  ✓ Best CV Score   : {grid_search.best_score_:.4f} ({grid_search.best_score_ * 100:.2f}%)")

best_svc = grid_search.best_estimator_


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — MODEL EVALUATION & PERFORMANCE METRICS                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

section_header("SECTION 8 — MODEL EVALUATION & PERFORMANCE METRICS")

# ── 8.1 Predictions ─────────────────────────────────────────────────────────
y_pred = best_svc.predict(X_test_scaled)
y_proba = best_svc.predict_proba(X_test_scaled)[:, 1]

# ── 8.2 Classification Report ───────────────────────────────────────────────
sub_header("8.2 Classification Report")
print(classification_report(y_test, y_pred, target_names=["Did Not Survive", "Survived"]))

# ── 8.3 Key Metrics ─────────────────────────────────────────────────────────
sub_header("8.3 Key Performance Metrics")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
avg_prec = average_precision_score(y_test, y_proba)

print(f"  Accuracy          : {acc:.4f}  ({acc * 100:.2f}%)")
print(f"  Precision         : {prec:.4f}")
print(f"  Recall            : {rec:.4f}")
print(f"  F1-Score          : {f1:.4f}")
print(f"  ROC-AUC           : {roc_auc:.4f}")
print(f"  Average Precision : {avg_prec:.4f}")

# ── 8.4 Cross-Validation Scores ─────────────────────────────────────────────
sub_header("8.4 Cross-Validation Scores (5-Fold)")
cv_scores = cross_val_score(
    best_svc, X_train_scaled, y_train,
    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    scoring="accuracy",
)
print(f"  Fold scores : {cv_scores.round(4)}")
print(f"  Mean ± Std  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Plot 8: Confusion Matrix ────────────────────────────────────────────────
sub_header("Plot 8: Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
sns.heatmap(
    cm, annot=True, fmt="d", cmap="RdYlGn", linewidths=2, linecolor="#1a1a2e",
    xticklabels=["Did Not Survive", "Survived"],
    yticklabels=["Did Not Survive", "Survived"],
    ax=ax, annot_kws={"size": 18, "fontweight": "bold"},
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", color="white", pad=15)
ax.set_xlabel("Predicted Label", fontsize=13, color="white")
ax.set_ylabel("True Label", fontsize=13, color="white")
ax.tick_params(colors="white", labelsize=11)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors="white")
fig.tight_layout()
save_fig(fig, "08_confusion_matrix.png")

# ── Plot 9: ROC Curve ───────────────────────────────────────────────────────
sub_header("Plot 9: ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(8, 6), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
ax.plot(fpr, tpr, color=PALETTE_SURVIVAL[1], linewidth=2.5,
        label=f"SVC (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="#E74C3C", linestyle="--", linewidth=1.5,
        label="Random Classifier", alpha=0.7)
ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE_SURVIVAL[1])
ax.set_title("Receiver Operating Characteristic (ROC) Curve",
             fontsize=16, fontweight="bold", color="white", pad=15)
ax.set_xlabel("False Positive Rate", fontsize=12, color="white")
ax.set_ylabel("True Positive Rate", fontsize=12, color="white")
ax.legend(fontsize=11, loc="lower right", framealpha=0.3)
ax.tick_params(colors="white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444")
ax.spines["bottom"].set_color("#444")
fig.tight_layout()
save_fig(fig, "09_roc_curve.png")




# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — FEATURE IMPORTANCE ANALYSIS                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

section_header("SECTION 9 — FEATURE IMPORTANCE ANALYSIS")

sub_header("9.1 Permutation Importance (on Test Set)")

perm_result = permutation_importance(
    best_svc, X_test_scaled, y_test,
    n_repeats=30, random_state=RANDOM_STATE, scoring="accuracy",
)

perm_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance Mean": perm_result.importances_mean,
    "Importance Std": perm_result.importances_std,
}).sort_values("Importance Mean", ascending=False)

print(perm_df.to_string(index=False))

# ── Plot 16: Feature Importance ──────────────────────────────────────────────
sub_header("Plot 10: Permutation Feature Importance")

perm_sorted = perm_df.sort_values("Importance Mean", ascending=True)
fig, ax = plt.subplots(figsize=(10, 7), facecolor="#1a1a2e")
ax.set_facecolor("#1a1a2e")
colors_imp = sns.color_palette("viridis", len(perm_sorted))
ax.barh(
    perm_sorted["Feature"], perm_sorted["Importance Mean"],
    xerr=perm_sorted["Importance Std"], color=colors_imp,
    edgecolor="white", linewidth=0.5, capsize=3,
)
ax.set_title("Permutation Feature Importance (SVC)",
             fontsize=16, fontweight="bold", color="white", pad=15)
ax.set_xlabel("Mean Accuracy Decrease", fontsize=12, color="white")
ax.set_ylabel("Feature", fontsize=12, color="white")
ax.tick_params(colors="white", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444")
ax.spines["bottom"].set_color("#444")
fig.tight_layout()
save_fig(fig, "10_feature_importance.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10 — SUMMARY & CONCLUSIONS                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

section_header("SECTION 10 — SUMMARY & CONCLUSIONS")

top5 = perm_df.head(5)["Feature"].tolist()

print(f"\n  Dataset: {df.shape[0]} records | {X.shape[1]} features | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"\n  Best SVC Parameters: {grid_search.best_params_}")
print(f"\n  Performance Metrics:")
print(f"    Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")
print(f"    F1-Score  : {f1:.4f}")
print(f"    ROC-AUC   : {roc_auc:.4f}")
print(f"    CV Score  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\n  Top 5 Predictive Features:")
for i, feat in enumerate(top5, 1):
    print(f"    {i}. {feat}")
print(f"\n  Key Insights:")
print(f"    - Women had much higher survival rates (women and children first).")
print(f"    - 1st class passengers survived at higher rates than 2nd/3rd class.")
print(f"    - Small families (2-4) survived more than solo or large families.")
print(f"    - Children with families had better survival odds.")

print(f"\n  All {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])} visualizations saved to: ./{OUTPUT_DIR}/")
print("\n" + "=" * 80)
print("  Analysis complete.")
print("=" * 80)

# Assignment No. 3: Exploratory Data Analysis (EDA)
# Dataset: UCI Heart Disease (Cleveland subset)
# Run this notebook in Google Colab. Internet access is required to fetch the dataset.

# In Google Colab, install the dependency first with: !pip -q install ucimlrepo

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# 1. Load dataset
heart = fetch_ucirepo(id=45)
X = heart.data.features.copy()
y = heart.data.targets.copy()

df = X.copy()
df["num"] = y.iloc[:, 0]

# The original target uses 0 for no disease and 1-4 for presence.
# Convert it into a binary target: 0 = no disease, 1 = disease present.
df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype("Int64")
df.drop(columns=["num"], inplace=True)

print("Dataset shape (rows, columns):", df.shape)
print("\\nFirst five rows:")
display(df.head())
print("\\nData types:")
display(df.dtypes.to_frame("dtype"))
print("\\nSummary statistics:")
display(df.describe(include="all").T)
print("\\nMissing values per column:")
display(df.isna().sum().sort_values(ascending=False).to_frame("missing_count"))
print("\\nDuplicate rows:", df.duplicated().sum())

# Friendly feature descriptions for chart labels
descriptions = {
    "age": "Age (years)",
    "sex": "Sex (0=female, 1=male)",
    "cp": "Chest pain type (coded)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar >120 (coded)",
    "restecg": "Resting ECG result (coded)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (0=no, 1=yes)",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Peak exercise ST slope (coded)",
    "ca": "Major vessels (0-3)",
    "thal": "Thal test result (coded)",
    "target": "Heart disease (0=no, 1=yes)"
}
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_features = [c for c in numeric_cols if c != "target"]
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# 2. Univariate EDA: numerical histograms and density plots
for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
    if col in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=col, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(descriptions.get(col, col))
        plt.tight_layout()
        plt.show()

# Boxplots help identify spread and possible unusual values
for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
    if col in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=col, ax=ax)
        ax.set_title(f"Boxplot of {col}")
        ax.set_xlabel(descriptions.get(col, col))
        plt.tight_layout()
        plt.show()

# Bar charts for categorical variables
for col in categorical_cols:
    if col in df.columns:
        fig, ax = plt.subplots()
        order = df[col].value_counts(dropna=False).index
        sns.countplot(data=df, x=col, order=order, ax=ax)
        ax.set_title(f"Counts for {col}")
        ax.set_xlabel(descriptions.get(col, col))
        ax.set_ylabel("Number of patients")
        plt.tight_layout()
        plt.show()

# 3. Target distribution and class balance
fig, ax = plt.subplots()
sns.countplot(data=df, x="target", order=[0, 1], ax=ax)
ax.set_title("Heart Disease Target Distribution")
ax.set_xlabel("Target (0 = no disease, 1 = disease present)")
ax.set_ylabel("Number of patients")
plt.tight_layout()
plt.show()

target_counts = df["target"].value_counts(dropna=False).sort_index()
target_percent = df["target"].value_counts(normalize=True, dropna=False).sort_index() * 100
print("Target counts:")
display(pd.DataFrame({"count": target_counts, "percent": target_percent.round(2)}))

# 4. Bivariate EDA: numerical features compared by target
for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
    if col in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="target", y=col, ax=ax)
        ax.set_title(f"{col} by heart disease target")
        ax.set_xlabel("Target (0 = no disease, 1 = disease present)")
        ax.set_ylabel(descriptions.get(col, col))
        plt.tight_layout()
        plt.show()

# Categorical features vs target: grouped bar charts showing within-category proportions
for col in ["sex", "cp", "exang", "slope", "thal"]:
    if col in df.columns:
        cross = pd.crosstab(df[col], df["target"], normalize="index")
        cross.plot(kind="bar", stacked=True, figsize=(8, 5))
        plt.title(f"Heart disease proportion by {col}")
        plt.xlabel(descriptions.get(col, col))
        plt.ylabel("Proportion within category")
        plt.legend(["No disease", "Disease present"], title="Target")
        plt.tight_layout()
        plt.show()

# Scatter plot for two numerical variables, coloured by target
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="age", y="thalach", hue="target", alpha=0.8, ax=ax)
ax.set_title("Age vs Maximum Heart Rate by Target")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Maximum heart rate achieved")
plt.tight_layout()
plt.show()

# 5. Correlation heatmap for numerical variables
corr = df[numeric_cols].corr(numeric_only=True)
plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()
plt.show()

# Print the strongest correlations with target, excluding target itself
target_corr = corr["target"].drop("target").dropna().sort_values(key=abs, ascending=False)
print("Features most correlated with target (Pearson correlation):")
display(target_corr.to_frame("correlation"))

# 6. Multivariate EDA: combine several features
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="age", y="thalach", hue="target", style="sex", alpha=0.8, ax=ax)
ax.set_title("Age, Maximum Heart Rate, Sex, and Target")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
sns.boxplot(data=df, x="cp", y="oldpeak", hue="target", ax=ax)
ax.set_title("Oldpeak by Chest Pain Type and Target")
ax.set_xlabel("Chest pain type (coded)")
ax.set_ylabel("ST depression (oldpeak)")
plt.tight_layout()
plt.show()

# Pairplot of selected numerical features
pair_cols = [c for c in ["age", "trestbps", "chol", "thalach", "oldpeak", "target"] if c in df.columns]
sns.pairplot(df[pair_cols].dropna(), hue="target", corner=True, diag_kind="hist")
plt.show()

# 7. Patterns, trends, and possible anomalies
print("Numerical feature ranges and quartiles:")
display(df[numeric_features].describe().T[["min", "25%", "50%", "75%", "max"]])

# IQR-based potential outlier counts. These are flags for review, not automatic errors.
outlier_rows = []
for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
    if col in df.columns:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = ((s < lower) | (s > upper)).sum()
        outlier_rows.append([col, lower, upper, int(count)])
outlier_table = pd.DataFrame(outlier_rows, columns=["feature", "IQR_lower_bound", "IQR_upper_bound", "potential_outliers"])
display(outlier_table)

# 8. Key findings: generate data-based summaries
print("\\nKEY SUMMARY VALUES")
print("Total rows:", len(df))
print("Total columns including target:", df.shape[1])
print("Missing cells:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("Target prevalence (% disease present):", round(df["target"].mean() * 100, 2))
for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
    if col in df.columns:
        print(f"{col}: median={df[col].median():.2f}, mean={df[col].mean():.2f}")
print("\\nNote: This dataset is for learning and research, not for diagnosing a real person.")

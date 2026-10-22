
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed.cleveland.data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load the Cleveland heart disease dataset
# --------------------------------------------------

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place processed.cleveland.data inside the data folder."
    )

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=columns,
    na_values=["?", " ?"],
    skipinitialspace=True,
)

# Convert all columns to numeric values.
# Invalid values become missing values.
for column in columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# The target must be present.
df = df.dropna(subset=["num"]).copy()

# Remove rows with unexpected target values.
df = df[df["num"].isin([0, 1, 2, 3, 4])].copy()

# Target:
# 0 = absence of heart disease in the dataset
# 1 = presence of heart disease in the dataset
df["target"] = (df["num"] > 0).astype(int)

# --------------------------------------------------
# 3. Select model input features
# --------------------------------------------------

features = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

X = df[features].copy()
y = df["target"].copy()

# Categorical features should not be treated as
# continuous measurements.
categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

numeric_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

print("\n" + "=" * 55)
print("HEARTCARE - HEART DISEASE CLASSIFICATION")
print("=" * 55)

print(f"\nDataset records: {len(df)}")
print(f"Number of input features: {len(features)}")

print("\nTarget distribution:")
print(y.value_counts().sort_index().rename(
    index={0: "Class 0", 1: "Class 1"}
))

print("\nMissing values before preprocessing:")
print(X.isnull().sum()[X.isnull().sum() > 0])

# --------------------------------------------------
# 4. Split data into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# 5. Prepare preprocessing pipelines
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

# --------------------------------------------------
# 6. Define three ML algorithms
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
    ),
}

trained_models = {}
results = []

# --------------------------------------------------
# 7. Train and evaluate each model
# --------------------------------------------------

for model_name, classifier in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    trained_models[model_name] = pipeline

    results.append(
        {
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
        }
    )

# --------------------------------------------------
# 8. Compare model performance
# --------------------------------------------------

results_df = pd.DataFrame(results)

# Choose by F1 score, using recall as the tie-breaker.
# This is a development choice, not clinical validation.
results_df = results_df.sort_values(
    by=["F1 Score", "Recall", "Accuracy"],
    ascending=False,
).reset_index(drop=True)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

# --------------------------------------------------
# 9. Evaluate the selected model
# --------------------------------------------------

final_predictions = best_model.predict(X_test)

print("\n" + "=" * 55)
print(f"SELECTED MODEL: {best_model_name}")
print("=" * 55)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "Class 0: No disease recorded",
            "Class 1: Disease recorded",
        ],
        zero_division=0,
    )
)

# --------------------------------------------------
# 10. Save the model and evaluation results
# --------------------------------------------------

model_path = MODEL_DIR / "heartcare_model.joblib"
results_path = MODEL_DIR / "model_comparison.csv"
info_path = MODEL_DIR / "model_info.json"

joblib.dump(best_model, model_path)
results_df.to_csv(results_path, index=False)

model_info = {
    "model_name": best_model_name,
    "features": features,
    "target_mapping": {
        "0": "No heart disease recorded in dataset",
        "1": "Heart disease recorded in dataset",
    },
    "positive_class": 1,
    "dataset": "UCI Cleveland Heart Disease dataset",
    "test_size": 0.20,
    "random_state": 42,
    "metrics": results_df.iloc[0].to_dict(),
}

with open(info_path, "w", encoding="utf-8") as file:
    json.dump(model_info, file, indent=4)

print("\n" + "=" * 55)
print("HEARTCARE MODEL TRAINING COMPLETED")
print("=" * 55)

print(f"\nSelected model: {best_model_name}")
print(f"Saved model: {model_path}")
print(f"Saved comparison: {results_path}")
print(f"Saved model information: {info_path}")

print(
    "\nNote: This model is an educational classification model. "
    "Its predictions are not a clinical diagnosis or a validated "
    "estimate of an individual's future cardiovascular risk."
)

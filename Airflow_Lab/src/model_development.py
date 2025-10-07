# src/model_development.py
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report


# ---------------------------
# Config / Paths
# ---------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "diabetes.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")


# Columns in the diabetes dataset
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# Columns where zeros should be treated as missing
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# ---------------------------
# Public API (used by Airflow)
# ---------------------------
def load_data():
    """
    Loads the Pima Indians Diabetes dataset from data/diabetes.csv.
    Returns:
        pd.DataFrame
    """
    df = pd.read_csv(DATA_PATH)
    _validate_diabetes_schema(df)
    return df


def data_preprocessing(data: pd.DataFrame):
    """
    Splits into train/test. We keep preprocessing inside the model pipeline,
    so this function only splits and returns raw features/labels.
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = data[FEATURES].copy()
    y = data[TARGET].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train.values, y_test.values


def build_model(data, filename: str):
    """
    Trains a Logistic Regression pipeline and saves it as <PROJECT_ROOT>/model/<filename>.
    Args:
        data: tuple (X_train, X_test, y_train, y_test)
        filename: name of the pickle file, e.g. "model_diabetes.sav"
    """
    X_train, X_test, y_train, y_test = data

    # Build pipeline: zero->NaN, impute median, standardize, logistic regression
    pipeline = Pipeline(
        steps=[
            ("zero_to_nan", FunctionTransformer(_zero_to_nan, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"[MODEL] ROC AUC: {auc:.4f}")
        print("[MODEL] Classification report:\n", classification_report(y_test, y_pred, digits=4))
    except Exception as e:
        print(f"[MODEL] Evaluation warning: {e}")

    # Persist
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[MODEL] Saved pipeline to: {model_path}")


def load_model(data, filename: str):
    """
    Loads the saved pipeline and evaluates it on the provided test split.
    Returns:
        int: first prediction (kept for backward compatibility with your previous code).
    """
    X_train, X_test, y_train, y_test = data
    model_path = os.path.join(MODEL_DIR, filename)

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    score = pipeline.score(X_test, y_test)
    print(f"[MODEL] Accuracy on test data: {score:.4f}")

    preds = pipeline.predict(X_test)
    return int(preds[0])


# ---------------------------
# Helpers
# ---------------------------
def _zero_to_nan(X: pd.DataFrame) -> pd.DataFrame:
    """
    Converts physiologically impossible zeros to NaN for specified columns.
    Works on DataFrame input; if ndarray is passed, converts to DataFrame using FEATURES.
    """
    if not isinstance(X, pd.DataFrame):
        # Convert ndarray to DataFrame with known feature order
        X = pd.DataFrame(X, columns=FEATURES)

    X = X.copy()
    for col in ZERO_AS_MISSING:
        if col in X.columns:
            X.loc[X[col] == 0, col] = np.nan
    return X


def _validate_diabetes_schema(df: pd.DataFrame):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset at {DATA_PATH} is missing required columns: {missing}. "
            f"Expected features: {FEATURES} and target: '{TARGET}'."
        )


# ---------------------------
# Local test run
# ---------------------------
if __name__ == "__main__":
    df_ = load_data()
    splits_ = data_preprocessing(df_)
    build_model(splits_, "model_diabetes.sav")
    _ = load_model(splits_, "model_diabetes.sav")

# src/train.py
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

TOP_K = 6  # change if you want more/less features in the minimal model

def main():
    ds = load_wine(as_frame=False)          # no pandas required
    X, y = ds.data, ds.target

    # Clean the JSON-unfriendly feature name
    feature_names = list(ds.feature_names)
    feature_names = [
        ("od280_od315_of_diluted_wines" if fn == "od280/od315 of diluted wines" else fn)
        for fn in feature_names
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Full model
    full_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", C=0.5, max_iter=2000))
    ])
    full_pipe.fit(X_train, y_train)
    full_acc = accuracy_score(y_test, full_pipe.predict(X_test))

    # Top-K feature selection
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    idx = np.argsort(mi)[::-1][:TOP_K]
    selected_names = [feature_names[i] for i in idx]

    # Minimal model
    Xk_train, Xk_test = X_train[:, idx], X_test[:, idx]
    min_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", C=0.5, max_iter=2000))
    ])
    min_pipe.fit(Xk_train, y_train)
    min_acc = accuracy_score(y_test, min_pipe.predict(Xk_test))

    model_dir = Path(__file__).resolve().parents[1] / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / "wine_model.pkl"

    joblib.dump({
        "full_model": full_pipe,
        "minimal_model": min_pipe,
        "feature_names": feature_names,      # all 13, in order
        "selected_names": selected_names,    # top-K, in order
        "class_names": list(ds.target_names),
        "metrics": {"full_acc": float(full_acc), "minimal_acc": float(min_acc), "k": TOP_K}
    }, out)

    print(f"Saved model to {out}")
    print(f"[FULL ] accuracy = {full_acc:.4f}")
    print(f"[MIN  ] accuracy = {min_acc:.4f}  (K={TOP_K})")
    print(f"Selected features: {selected_names}")

if __name__ == "__main__":
    main()

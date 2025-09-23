# Wine Classifier API (FastAPI + scikit‑learn)

Serve a scikit‑learn model as a **FastAPI** service. The project trains a classifier on the classic **Wine** dataset, saves the model with `joblib`, and exposes a single prediction endpoint.

---

## Features

* 🚀 FastAPI app with a **single POST** endpoint: `POST /predict`
* 🧠 Model: scikit‑learn pipeline (scaler + logistic regression)
* 💾 Serialized bundle saved to `model/wine_model.pkl`
* 🧰 Optionally exposes `GET /` for health & metadata (feature names, classes)

---

## Project Structure

```
FAST_API/
├── assets/                     
├── model/
│   └── wine_model.pkl          
├── src/
│   ├── __init__.py            
│   ├── data.py                
│   ├── main.py                
│   ├── predict.py             
│   └── train.py               
├── README.md                  
└── requirements.txt           
```

---


## Quickstart

### 1) Create & activate a virtual environment

**macOS/Linux**

```bash
python3 -m venv fastapi_lab1_env
source fastapi_lab1_env/bin/activate
python -m pip install --upgrade pip
```


### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Train the model

Run from the **project root** (same folder that contains `src/`):

```bash
python src/train.py
```

This will create/update `model/wine_model.pkl` and print accuracy + selected features.

### 4) Serve the API

Run from the **project root** so package imports work:

```bash
uvicorn src.main:app --reload
```

Now open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API

### Health & metadata

`GET /`

* Returns: available feature names (`full_features`, `minimal_features`) and `classes`.

### Predict

`POST /predict`

* **Request body** (`WineData`):

  ```json
  {
    "alcohol": 13.2,
    "malic_acid": 1.8,
    "ash": 2.36,
    "alcalinity_of_ash": 19.0,
    "magnesium": 100.0,
    "total_phenols": 2.6,
    "flavanoids": 2.9,
    "nonflavanoid_phenols": 0.30,
    "proanthocyanins": 1.7,
    "color_intensity": 5.0,
    "hue": 1.05,
    "od280_od315_of_diluted_wines": 3.05,
    "proline": 1100.0
  }
  ```
* **Response** (`WineProbaResponse`):

  ```json
  {
    "probabilities": [0.02, 0.95, 0.03],
    "class_names": ["class_0", "class_1", "class_2"],
    "predicted_label": 1
  }
  ```

> The server is robust to the slash‑named feature; either key works:
> `"od280/od315 of diluted wines"` **or** `"od280_od315_of_diluted_wines"`.

---

## cURL Examples

**Full predict**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 13.2,
    "malic_acid": 1.8,
    "ash": 2.36,
    "alcalinity_of_ash": 19.0,
    "magnesium": 100.0,
    "total_phenols": 2.6,
    "flavanoids": 2.9,
    "nonflavanoid_phenols": 0.30,
    "proanthocyanins": 1.7,
    "color_intensity": 5.0,
    "hue": 1.05,
    "od280_od315_of_diluted_wines": 3.05,
    "proline": 1100.0
  }'
```

---

## How it works (internals)

* **Training** (`src/train.py`):

  * Loads the Wine dataset (arrays only; no pandas required).
  * Cleans the JSON‑unfriendly feature name (`od280/od315 of diluted wines` → `od280_od315_of_diluted_wines`).
  * Builds two pipelines (full & minimal) using `StandardScaler` + `LogisticRegression(solver="liblinear", C=0.5, max_iter=2000)`.
  * Saves the bundle (models + metadata) to `model/wine_model.pkl`.
* **Serving** (`src/main.py`):

  * Loads the bundle once at startup.
  * Exposes `POST /predict` and `GET /`.
  * Accepts either variant of the `od280…` feature key via an alias map.

---

# 🩺 README.md --- Airflow Lab 2 -- Diabetes Prediction Pipeline

## 📘 Overview

This project implements a **machine-learning pipeline** for **Diabetes
Prediction** using **Logistic Regression**, orchestrated with **Apache
Airflow** and served through a **Flask API**.

The pipeline:

-   Loads and preprocesses the **Pima Indians Diabetes Dataset**
-   Trains and saves a `LogisticRegression` model (`model_diabetes.sav`)
-   Sends a notification email upon completion
-   Triggers a separate **Flask DAG** that displays the Airflow DAG's
    run status (`success` / `failure`)

------------------------------------------------------------------------

## 🧩 Project Structure

    project_root/
    ├── dags/
    │   ├── main.py                 # Airflow DAG for training & notifications
    │   └── Airflow_Lab2_Flask.py   # Flask API DAG for status monitoring
    ├── src/
    │   └── model_development.py    # ML pipeline code for training and loading model
    ├── data/
    │   └── diabetes.csv            # Pima Indians Diabetes dataset
    ├── model/
    │   └── model_diabetes.sav      # Saved Logistic Regression model
    ├── templates/
    │   ├── success.html            # Flask success page
    │   └── failure.html            # Flask failure page
    └── README.md                   # This file

------------------------------------------------------------------------

## ⚙️ Pipeline Workflow ( Airflow Lab 2 )

The **main Airflow DAG** (`main.py`) executes the following tasks:

  ---------------------------------------------------------------------------
  Step           Task ID                      Description
  -------------- ---------------------------- -------------------------------
  1              owner_verification           Simple bash task to start the
                                              DAG

  2              load_data_task               Loads the diabetes dataset
                                              (`data/diabetes.csv`)

  3              data_preprocessing_task      Splits into train/test sets

  4              separate_data_outputs_task   Extracts train/test arrays for
                                              next tasks

  5              build_save_model_task        Trains and saves the Logistic
                                              Regression model

  6              load_model_task              Reloads the saved model to
                                              validate integrity

  7              send_email                   Sends completion notification
                                              email

  8              trigger_flask_api            Triggers the Flask DAG
                                              (`Airflow_Lab2_Flask`)
  ---------------------------------------------------------------------------

------------------------------------------------------------------------

## 🧠 Machine Learning Details

-   **Dataset:** [Pima Indians Diabetes
    (UCI)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
-   **Features:**
    -   Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
        BMI, DiabetesPedigreeFunction, Age
-   **Target:** `Outcome` ( 0 = No Diabetes, 1 = Diabetes )
-   **Model:**
    `LogisticRegression(class_weight="balanced", max_iter=1000)`
-   **Preprocessing:**
    -   Zeros in medical fields → NaN
    -   Median imputation
    -   Standard scaling
-   **Evaluation Metrics:** Accuracy & ROC-AUC on test data

------------------------------------------------------------------------

## 🌐 Flask API (DAG Airflow_Lab2_Flask)

Launches a Flask server after the training pipeline finishes.

Provides simple status pages using HTML templates.

  Route        Purpose
  ------------ -----------------------------------------------------------
  `/`          Redirects to `/success` or `/failure` based on DAG status
  `/success`   Displays success message if DAG completed successfully
  `/failure`   Shows error message if DAG failed or has not run
  `/health`    Returns JSON health status for monitoring

**Runs on:** `http://localhost:5555`

------------------------------------------------------------------------

## 📧 Email Notification

When the model finishes training, Airflow automatically sends an email
(using `EmailOperator`) to the configured address:

`harrishebifrancis@gmail.com`

### To customize:

Edit the `send_email` task in `dags/main.py`:

``` python
to='youremail@example.com'
```

------------------------------------------------------------------------

## 🚀 Running the Pipeline

### 1. Start Airflow

``` bash
airflow db init
airflow webserver -p 8080
airflow scheduler
```

### 2. Access the UI → <http://localhost:8080>

### 3. Trigger the main DAG

**In UI:** DAGs → Airflow_Lab2 → Trigger DAG

**Or via CLI:**

``` bash
airflow dags trigger Airflow_Lab2
```

After training completes, Airflow will:

-   Send the notification email
-   Trigger `Airflow_Lab2_Flask` to launch the Flask app

------------------------------------------------------------------------

## 🧰 Requirements

    apache-airflow>=2.7
    flask
    pandas
    numpy
    scikit-learn>=1.1
    joblib
    email-validator

------------------------------------------------------------------------

## 🧪 Testing the Flask App Manually

If you want to run the Flask status page stand-alone:

``` bash
python dags/Airflow_Lab2_Flask.py
```

Then visit:

✅ <http://localhost:5555/success>\
❌ <http://localhost:5555/failure>

------------------------------------------------------------------------

## 🧾 Outputs

-   **Model Artifact:** `model/model_diabetes.sav`
-   **Logs:** Airflow UI → Task Instances Logs
-   **Email:** Notification of pipeline completion
-   **Web:** Flask status page (`/success` / `/failure`)

------------------------------------------------------------------------

## 📚 Future Improvements

-   Add model evaluation and ROC curve logging
-   Store artifacts to S3 or GCS
-   Containerize Airflow + Flask with Docker Compose
-   Integrate MLflow for model tracking

------------------------------------------------------------------------

## 🧑‍💻 Maintainer

**Harrish Ebi Francis**\
✉️ <harrishebifrancis@gmail.com>

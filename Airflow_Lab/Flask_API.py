from airflow.models.dagrun import DagRun
from airflow.utils.state import State
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from flask import Flask, jsonify, redirect, render_template
import time
import os

# ------------------------------------------------------------------------
# Default arguments
# ------------------------------------------------------------------------
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 1),
    'retries': 0,
    'retry_delay': timedelta(minutes=1)
}

# ------------------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")

# ------------------------------------------------------------------------
# Helper Function: Check DAG run status
# ------------------------------------------------------------------------
def check_dag_status():
    """
    Checks whether the latest Airflow DAG run for 'Airflow_Lab2'
    completed successfully.
    Returns:
        True if success, False otherwise.
    """
    dag_id = "Airflow_Lab2"
    try:
        dag_runs = DagRun.find(dag_id=dag_id)
        if not dag_runs:
            print("[FLASK] No DAG runs found for Airflow_Lab2")
            return False
        latest_run = sorted(dag_runs, key=lambda x: x.execution_date)[-1]
        print(f"[FLASK] Latest DAG run state: {latest_run.state}")
        return latest_run.state == State.SUCCESS
    except Exception as e:
        print(f"[FLASK] Error checking DAG status: {e}")
        return False

# ------------------------------------------------------------------------
# Flask Routes
# ------------------------------------------------------------------------
@app.route("/")
def index():
    """
    Root route: Redirects to /success or /failure based on DAG status.
    """
    status = check_dag_status()
    if status:
        return redirect("/success")
    else:
        return redirect("/failure")


@app.route("/success")
def success():
    """
    Success route: Display confirmation when DAG completed successfully.
    """
    return render_template(
        "success.html",
        message="✅ Airflow_Lab2 DAG completed successfully! The Diabetes model is ready."
    )


@app.route("/failure")
def failure():
    """
    Failure route: Display error message if DAG failed or hasn't run yet.
    """
    return render_template(
        "failure.html",
        message="❌ The Airflow_Lab2 DAG failed or has not completed yet. Please check Airflow logs."
    )


@app.route("/health")
def health():
    """
    Simple health check endpoint for monitoring.
    """
    return jsonify({"status": "ok", "service": "Airflow_Lab2_Flask"}), 200


# ------------------------------------------------------------------------
# Function to start Flask server
# ------------------------------------------------------------------------
def start_flask_app():
    """
    Starts the Flask web application. Intended to be called by Airflow.
    """
    print("[FLASK] Starting Flask app on port 5555...")
    app.run(host="0.0.0.0", port=5555, debug=False)


# ------------------------------------------------------------------------
# Airflow DAG Definition
# ------------------------------------------------------------------------
flask_api_dag = DAG(
    dag_id="Airflow_Lab2_Flask",
    default_args=default_args,
    description="Flask API DAG to display Airflow_Lab2 status",
    schedule_interval=None,   # Triggered manually from main DAG
    catchup=False,
    tags=["flask_api", "status_monitor"],
)

# ------------------------------------------------------------------------
# Airflow Task: Start Flask API
# ------------------------------------------------------------------------
start_flask_api = PythonOperator(
    task_id="start_flask_api",
    python_callable=start_flask_app,
    dag=flask_api_dag,
)

# ------------------------------------------------------------------------
# DAG Task Dependencies
# ------------------------------------------------------------------------
start_flask_api

# ------------------------------------------------------------------------
# Direct Run (for local testing)
# ------------------------------------------------------------------------
if __name__ == "__main__":
    start_flask_app()

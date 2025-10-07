from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow import configuration as conf

from datetime import datetime, timedelta
from src.model_development import load_data, data_preprocessing, build_model, load_model

# Enable pickle support for XCom (for sklearn objects)
conf.set('core', 'enable_xcom_pickling', 'True')

# ------------------------------------------------------------------------
# Default arguments
# ------------------------------------------------------------------------
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=2),
}

# ------------------------------------------------------------------------
# DAG definition
# ------------------------------------------------------------------------
dag = DAG(
    'Airflow_Lab2',
    default_args=default_args,
    description='Airflow DAG for Diabetes Prediction using Logistic Regression',
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'diabetes', 'airflow'],
    owner_links={"Ramin Mohammadi": "https://github.com/raminmohammadi/MLOps/"}
)

# ------------------------------------------------------------------------
# Task 1: Owner task (simple bash)
# ------------------------------------------------------------------------
owner_task = BashOperator(
    task_id="owner_verification",
    bash_command="echo 'DAG started by Francis'",
    dag=dag,
)

# ------------------------------------------------------------------------
# Task 2: Load the diabetes dataset
# ------------------------------------------------------------------------
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# ------------------------------------------------------------------------
# Task 3: Data preprocessing (split train/test)
# ------------------------------------------------------------------------
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],  # uses output from load_data_task
    dag=dag,
)

# ------------------------------------------------------------------------
# Task 4: Separate outputs (keep this to maintain your structure)
# ------------------------------------------------------------------------
def separate_data_outputs(**kwargs):
    ti = kwargs['ti']
    X_train, X_test, y_train, y_test = ti.xcom_pull(task_ids='data_preprocessing_task')
    return X_train, X_test, y_train, y_test

separate_data_outputs_task = PythonOperator(
    task_id='separate_data_outputs_task',
    python_callable=separate_data_outputs,
    dag=dag,
)

# ------------------------------------------------------------------------
# Task 5: Build and save Logistic Regression model
# ------------------------------------------------------------------------
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_model,
    op_args=[separate_data_outputs_task.output, "model_diabetes.sav"],
    dag=dag,
)

# ------------------------------------------------------------------------
# Task 6: Load and evaluate the saved model
# ------------------------------------------------------------------------
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model,
    op_args=[separate_data_outputs_task.output, "model_diabetes.sav"],
    dag=dag,
)

# ------------------------------------------------------------------------
# Task 7: Send email notification
# ------------------------------------------------------------------------
send_email = EmailOperator(
    task_id='send_email',
    to='harrishebifrancis@gmail.com',
    subject='Airflow Notification: Diabetes Model Pipeline Complete',
    html_content="""
    <h3>Airflow Pipeline Complete</h3>
    <p>The Diabetes Logistic Regression model has completed training and evaluation.</p>
    <p><strong>Model:</strong> model_diabetes.sav</p>
    <p><strong>Timestamp:</strong> {{ ds }}</p>
    <p>- Airflow Automation</p>
    """,
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,  # ensures email still sent if minor failures
    dag=dag,
)

# ------------------------------------------------------------------------
# Task 8: Trigger the Flask API DAG
# ------------------------------------------------------------------------
trigger_flask_dag = TriggerDagRunOperator(
    task_id='trigger_flask_api',
    trigger_rule=TriggerRule.ALL_DONE,  # triggers regardless of success/failure
    trigger_dag_id='Airflow_Lab2_Flask',
    dag=dag,
)

# ------------------------------------------------------------------------
# Dependencies (order of execution)
# ------------------------------------------------------------------------
(
    owner_task
    >> load_data_task
    >> data_preprocessing_task
    >> separate_data_outputs_task
    >> build_save_model_task
    >> load_model_task
    >> send_email
    >> trigger_flask_dag
)

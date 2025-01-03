from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
import os
import glob
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.ui.workspace.cloud import CloudWorkspace
import logging

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)

# Variables Airflow
EVIDENTLY_CLOUD_TOKEN = Variable.get("EVIDENTLY_CLOUD_TOKEN") 
EVIDENTLY_CLOUD_PROJECT_ID = Variable.get("EVIDENTLY_CLOUD_PROJECT_ID")

# Chemins des répertoires
DATA_DIR = "/opt/airflow/data"
REFERENCE_DIR = os.path.join(DATA_DIR, "reference")
DATA_DRIFT_DIR = os.path.join(DATA_DIR, "data-drift")
REFERENCE_FILE = os.path.join(REFERENCE_DIR, "covtype_reference_first100.csv")

# Colonnes à analyser
COLUMNS_TO_ANALYZE = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

def _load_files(data_logs_filename):
    reference = pd.read_csv(REFERENCE_FILE)
    data_logs = pd.read_csv(data_logs_filename)
    return reference, data_logs

def detect_file(**context):
    data_logs_list = glob.glob(os.path.join(DATA_DRIFT_DIR, "covtype_reference_update*.csv"))
    if not data_logs_list:
        return "no_file_found_task"
    data_logs_filename = max(data_logs_list, key=os.path.getctime)
    context["task_instance"].xcom_push(key="data_logs_filename", value=data_logs_filename)
    return "detect_data_drift_task"

def detect_data_drift(**context):
    """Produire un rapport de dérive des données avec Evidently Cloud"""

    # Initialiser la connexion au workspace Evidently Cloud
    ws = CloudWorkspace(
        token=EVIDENTLY_CLOUD_TOKEN,
        url="https://app.evidently.cloud"
    )

    project = ws.get_project(EVIDENTLY_CLOUD_PROJECT_ID)

    data_logs_filename = context["task_instance"].xcom_pull(key="data_logs_filename")
    logging.debug(f"Récupéré le fichier des logs: {data_logs_filename}")

    reference, data_logs = _load_files(data_logs_filename)

    # Validation des colonnes
    assert all(col in reference.columns for col in COLUMNS_TO_ANALYZE), "Colonnes manquantes dans les données de référence"
    assert all(col in data_logs.columns for col in COLUMNS_TO_ANALYZE), "Colonnes manquantes dans les logs"

    reference_filtered = reference[COLUMNS_TO_ANALYZE]
    data_logs_filtered = data_logs[COLUMNS_TO_ANALYZE]

    data_drift_report = Report(metrics=[DataDriftPreset()])
    logging.debug("Rapport de dérive créé.")

    try:
        data_drift_report.run(current_data=data_logs_filtered, reference_data=reference_filtered)
        logging.debug("Rapport de dérive exécuté avec succès.")

        ws.add_report(project.id, data_drift_report, include_data=True)
        logging.info("Rapport envoyé à Evidently Cloud.")

        # Log des métriques
        drift_results = data_drift_report.as_dict()
        logging.info(f"Résultats détaillés de la dérive : {drift_results}")

        # Rechercher la métrique DatasetDriftMetric
        dataset_drift_metric = next(
            (metric["result"] for metric in drift_results["metrics"] if metric["metric"] == "DatasetDriftMetric"),
            None
        )

        if not dataset_drift_metric:
            raise ValueError("Métrique 'DatasetDriftMetric' introuvable dans le rapport.")

        # Extraire le nombre de colonnes dérivées
        data_drift_summary = dataset_drift_metric.get("number_of_drifted_columns", 0)
        context["task_instance"].xcom_push(key="drift_summary", value=data_drift_summary)
        logging.info(f"Nombre de colonnes dérivées détectées : {data_drift_summary}")

        # Rechercher la métrique DataDriftTable pour les colonnes dérivées
        data_drift_table = next(
            (metric["result"] for metric in drift_results["metrics"] if metric["metric"] == "DataDriftTable"),
            None
        )

        if data_drift_table:
            drifted_columns = [
                col for col, details in data_drift_table.get("drift_by_columns", {}).items() if details["drift_detected"]
            ]
            context["task_instance"].xcom_push(key="drifted_columns", value=drifted_columns)
            logging.info(f"Colonnes dérivées : {', '.join(drifted_columns)}")

        # Décision basée sur la dérive détectée
        if data_drift_summary > 0:
            logging.info(f"Dérive détectée dans {data_drift_summary} colonnes.")
            context["task_instance"].xcom_push(key="drift_detected", value=True)
            return ['trigger_jenkins_task', 'prepare_email_content_task']
        else:
            logging.info("Aucune dérive détectée.")
            return 'no_drift_detected_task'

    except Exception as e:
        logging.error(f"Erreur lors de la génération du rapport de dérive: {e}")
        raise

def trigger_jenkins_retrain(**context):
    """Déclenche le retraining via Jenkins"""
    jenkins_url = "http://jenkins:8080"
    job_name = "cover_type_retrain"

    try:
        response = requests.post(
            f"{jenkins_url}/job/{job_name}/build",
            auth=(Variable.get("JENKINS_USER"), Variable.get("JENKINS_TOKEN"))
        )
        if response.status_code == 201:
            logging.info("Jenkins pipeline triggered successfully")
        else:
            logging.error(f"Failed to trigger Jenkins: {response.status_code}")
            raise Exception("Failed to trigger Jenkins pipeline")
    except Exception as e:
        logging.error(f"Error triggering Jenkins: {e}")
        raise

def prepare_email_content_task(**context):
    ti = context['ti']
    drift_summary = ti.xcom_pull(key='drift_summary', task_ids='detect_data_drift_task')
    drifted_columns = ti.xcom_pull(key='drifted_columns', task_ids='detect_data_drift_task') or []

    logging.info(f"Drift Summary reçu: {drift_summary}")
    logging.info(f"Colonnes dérivées : {drifted_columns}")

    if drift_summary == 0:
        subject = "Pas de drift détecté"
        body = "Aucun drift n'a été détecté dans les données analysées."
    else:
        subject = f"Drift détecté dans {drift_summary} colonnes"
        body = f"Le rapport de drift a détecté un drift dans les colonnes suivantes : {', '.join(drifted_columns)}."

    context['ti'].xcom_push(key='email_subject', value=subject)
    context['ti'].xcom_push(key='email_body', value=body)

def send_email_with_smtp(**context):
    ti = context['ti']

    subject = ti.xcom_pull(key='email_subject', task_ids='prepare_email_content_task')
    body = ti.xcom_pull(key='email_body', task_ids='prepare_email_content_task')

    if subject is None or body is None:
        raise ValueError("Le sujet ou le corps de l'email est manquant.")

    to_email = "xxx@gmail.com"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "xxx@gmail.com"

    smtp_password = Variable.get("gmail_password", default_var=None)
    if smtp_password is None:
        raise ValueError("Le mot de passe Gmail n'est pas défini dans les variables Airflow.")

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = to_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return f"Email envoyé à {to_email} avec succès !"
    except Exception as e:
        error_message = f"Erreur lors de l'envoi de l'e-mail : {str(e)}"
        print(error_message)
        raise Exception(error_message)

default_args = {
    'owner': 'RL',
    'start_date': datetime(2024, 10, 10),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'detect_data_drift_and_notify',
    default_args=default_args,
    description='Détecte la dérive des données et envoie une notification par email',
    schedule_interval=timedelta(days=1),
)

detect_file_task = BranchPythonOperator(
    task_id='detect_file_task',
    python_callable=detect_file,
    provide_context=True,
    dag=dag,
)

detect_data_drift_task = PythonOperator(
    task_id='detect_data_drift_task',
    python_callable=detect_data_drift,
    provide_context=True,
    dag=dag,
)

trigger_jenkins_task = PythonOperator(
    task_id='trigger_jenkins_task',
    python_callable=trigger_jenkins_retrain,
    provide_context=True,
    dag=dag,
)

prepare_email_content_task = PythonOperator(
    task_id='prepare_email_content_task',
    python_callable=prepare_email_content_task,
    provide_context=True,
    dag=dag,
)

send_email_task = PythonOperator(
    task_id='send_email_task',
    python_callable=send_email_with_smtp,
    provide_context=True,
    dag=dag,
)

no_file_found_task = DummyOperator(
    task_id='no_file_found_task',
    dag=dag,
)

detect_file_task >> [detect_data_drift_task, no_file_found_task]
detect_data_drift_task >> [trigger_jenkins_task, prepare_email_content_task]
prepare_email_content_task >> send_email_task
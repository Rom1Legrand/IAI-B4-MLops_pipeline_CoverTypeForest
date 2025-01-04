from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.ui.workspace.cloud import CloudWorkspace
import logging
import requests
import boto3

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)

# Variables Airflow
EVIDENTLY_CLOUD_TOKEN = Variable.get("EVIDENTLY_CLOUD_TOKEN") 
EVIDENTLY_CLOUD_PROJECT_ID = Variable.get("EVIDENTLY_CLOUD_PROJECT_ID")
S3_BUCKET = Variable.get("S3_BUCKET")

# Configuration S3
REFERENCE_FILE = 'covertype/reference/covtype_80.csv'
NEW_DATA_FILE = 'covertype/new_data/covtype_20.csv'

# Colonnes à analyser
COLUMNS_TO_ANALYZE = [
   "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
   "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
   "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
   "Horizontal_Distance_To_Fire_Points"
]

def _load_files():
   """Charger les fichiers depuis S3"""
   try:
       logging.info("Starting to load files from S3...")
       s3 = boto3.client('s3')
       
       logging.info(f"Loading reference file: {S3_BUCKET}/{REFERENCE_FILE}")
       ref_obj = s3.get_object(Bucket=S3_BUCKET, Key=REFERENCE_FILE)
       reference = pd.read_csv(ref_obj['Body'])
       logging.info(f"Reference file loaded, shape: {reference.shape}")
       
       logging.info(f"Loading new data file: {S3_BUCKET}/{NEW_DATA_FILE}")
       new_obj = s3.get_object(Bucket=S3_BUCKET, Key=NEW_DATA_FILE)
       new_data = pd.read_csv(new_obj['Body'])
       logging.info(f"New data file loaded, shape: {new_data.shape}")
       
       return reference, new_data
   except Exception as e:
       logging.error(f"Error loading files from S3: {str(e)}")
       raise

def trigger_jenkins():
   """Déclencher le retraining Jenkins"""
   jenkins_url = "http://jenkins:8080"
   job_name = "covertype_retrain"
   
   response = requests.post(
       f"{jenkins_url}/job/{job_name}/build",
       auth=(Variable.get("JENKINS_USER"), Variable.get("JENKINS_TOKEN"))
   )
   if response.status_code != 201:
       raise Exception(f"Failed to trigger Jenkins: {response.status_code}")
   logging.info("Jenkins pipeline triggered successfully")

def send_email(subject, body):
   """Envoyer un email de notification"""
   to_email = "dsgattaca@gmail.com"
   smtp_server = "smtp.gmail.com"
   smtp_port = 587
   smtp_user = "dsgattaca@gmail.com"
   
   smtp_password = Variable.get("gmail_password")
   
   msg = EmailMessage()
   msg.set_content(body)
   msg['Subject'] = subject
   msg['From'] = smtp_user
   msg['To'] = to_email
   
   with smtplib.SMTP(smtp_server, smtp_port) as server:
       server.starttls()
       server.login(smtp_user, smtp_password)
       server.send_message(msg)
   logging.info(f"Email sent to {to_email}")

def check_drift_and_notify(**context):
   """Fonction principale qui gère tout le processus"""
   try:
       # Vérifier le fichier S3
       s3 = boto3.client('s3')
       s3.head_object(Bucket=S3_BUCKET, Key=NEW_DATA_FILE)
       logging.info("File found in S3")
       
       # Charger et analyser les données
       reference, new_data = _load_files()
       
       # Configuration Evidently
       ws = CloudWorkspace(
           token=EVIDENTLY_CLOUD_TOKEN,
           url="https://app.evidently.cloud"
       )
       project = ws.get_project(EVIDENTLY_CLOUD_PROJECT_ID)
       
       # Validation et filtrage des colonnes
       reference_filtered = reference[COLUMNS_TO_ANALYZE]
       new_data_filtered = new_data[COLUMNS_TO_ANALYZE]
       
       # Analyse du drift
       data_drift_report = Report(metrics=[DataDriftPreset()])
       data_drift_report.run(current_data=new_data_filtered, reference_data=reference_filtered)
       
       # Envoi du rapport à Evidently Cloud
       ws.add_report(project.id, data_drift_report, include_data=True)
       
       # Analyse des résultats
       drift_results = data_drift_report.as_dict()
       dataset_drift_metric = next(
           (metric["result"] for metric in drift_results["metrics"] if metric["metric"] == "DatasetDriftMetric"),
           None
       )
       
       if not dataset_drift_metric:
           raise ValueError("Métrique DatasetDriftMetric introuvable")
       
       drift_summary = dataset_drift_metric.get("number_of_drifted_columns", 0)
       drifted_columns = []
       
       if drift_summary > 0:
           logging.info(f"Drift détecté dans {drift_summary} colonnes")
           trigger_jenkins()
           subject = f"Drift détecté dans {drift_summary} colonnes"
           body = f"Un drift a été détecté et le réentraînement a été lancé."
       else:
           logging.info("Aucun drift détecté")
           subject = "Pas de drift détecté"
           body = "L'analyse n'a détecté aucun drift dans les données."
       
       send_email(subject, body)
       
   except Exception as e:
       error_msg = f"Erreur dans le process: {str(e)}"
       logging.error(error_msg)
       send_email("Erreur dans l'analyse de drift", error_msg)
       raise

default_args = {
   'owner': 'RL',
   'start_date': datetime(2024, 10, 10),
   'email_on_failure': True,
   'email_on_retry': False,
   'retries': 1,
   'retry_delay': timedelta(minutes=5),
}

# DAG
dag = DAG(
   'drift_detection_pipeline',
   default_args=default_args,
   description='Pipeline de détection de drift et réentraînement',
   schedule_interval=None,
   catchup=False,
)

# Une seule tâche qui gère tout
check_drift_task = PythonOperator(
   task_id='check_drift_and_notify',
   python_callable=check_drift_and_notify,
   provide_context=True,
   dag=dag,
)
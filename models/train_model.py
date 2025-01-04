import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import boto3
import logging

logging.basicConfig(level=logging.INFO)

def train():
    try:
        # Configuration MLflow avec PostgreSQL directement
        tracking_uri = os.environ['NEON_DATABASE_URL'].replace('postgresql', 'postgresql+psycopg2')
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("forest_cover_type")
        logging.info("MLflow configuré")

        # Charger les données
        s3 = boto3.client('s3')
        bucket = os.environ['S3_BUCKET']
        data_obj = s3.get_object(Bucket=bucket, Key='covertype/new_data/covtype_20.csv')
        data = pd.read_csv(data_obj['Body'])
        logging.info(f"Données chargées : {data.shape[0]} échantillons")

        # Préparation des données
        X = data.drop('target', axis=1)
        y = data['target']

        # Entrainement
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)
            
            accuracy = model.score(X, y)
            logging.info(f"Accuracy: {accuracy}")

            mlflow.sklearn.log_model(model, "model")
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy)
            
    except Exception as e:
        logging.error(f"Erreur : {str(e)}")
        raise

if __name__ == "__main__":
    train()
import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import boto3
import logging
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO)

def train():
    try:
        # Configuration MLflow
        tracking_uri = os.environ['NEON_DATABASE_URL']
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("forest_cover_type")
        logging.info("MLflow configuré")

        # Charger les données
        s3 = boto3.client('s3')
        bucket = os.environ['S3_BUCKET']
        data_obj = s3.get_object(Bucket=bucket, Key='covertype/new_data/covtype_20.csv')
        data = pd.read_csv(data_obj['Body'])
        logging.info(f"Données chargées : {data.shape[0]} échantillons")

        # Vérifier et renommer la colonne cible si nécessaire
        target_column = 'target'
        if target_column not in data.columns:
            # Vérifier les autres noms possibles comme 'Cover_Type', 'class', etc.
            if 'Cover_Type' in data.columns:
                data = data.rename(columns={'Cover_Type': 'target'})
            # Ajouter d'autres conditions si nécessaire
        
        logging.info(f"Colonnes disponibles : {data.columns.tolist()}")

        # Préparation des données
        X = data.drop('target', axis=1)
        y = data['target']

        # Entrainement
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=10)
            model.fit(X, y)
            
            accuracy = model.score(X, y)
            f1 = f1_score(y, model.predict(X), average='weighted')  # weighted car problème multi-classe
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"F1 Score: {f1}")

            mlflow.sklearn.log_model(model, "model")
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)         
            
    except Exception as e:
        logging.error(f"Erreur : {str(e)}")
        raise

if __name__ == "__main__":
    train()
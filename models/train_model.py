import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import boto3

def train():
    # Configuration MLflow
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI']) # A VERIFIER
    mlflow.set_experiment("forest_cover_type")

    # Charger les données
    s3 = boto3.client('s3')
    data_obj = s3.get_object(Bucket='rom1', Key='covertype/new_data/covtype_20.csv')
    data = pd.read_csv(data_obj['Body'])

    # Préparation des données
    X = data.drop('target', axis=1)
    y = data['target']

    # Entrainement
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        
        # Log métriques et modèle
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", model.score(X, y))

if __name__ == "__main__":
    train()
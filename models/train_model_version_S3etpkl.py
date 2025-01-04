import os
import boto3
import pickle
import logging
import mlflow
from dotenv import load_dotenv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))  # Dossier 'models'
project_root = os.path.dirname(current_dir)  # Dossier racine du projet

# Chemins vers les fichiers de configuration
env_path = os.path.join(project_root, '.env')
secrets_path = os.path.join(project_root, '.secrets')

logger.info(f"Chargement des variables depuis {env_path}")
logger.info(f"Chargement des secrets depuis {secrets_path}")

# Chargement des variables d'environnement
load_dotenv(env_path)
load_dotenv(secrets_path)

# Verify AWS credentials
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not aws_access_key or not aws_secret_key:
    logger.error("AWS credentials manquants:")
    logger.error(f"AWS_ACCESS_KEY_ID: {'Présent' if aws_access_key else 'Manquant'}")
    logger.error(f"AWS_SECRET_ACCESS_KEY: {'Présent' if aws_secret_key else 'Manquant'}")
    exit(1)

# Create S3 client with credentials
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define required columns
PREDICTION_COLUMNS = [
    'Elevation', 'Horizontal_Distance_To_Roadways',
    'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Aspect', 'Wilderness_Area4',
    'Hillshade_Noon', 'Hillshade_3pm', 'Hillshade_9am', 'Slope',
    'Soil_Type22', 'Soil_Type10', 'Soil_Type4', 'Soil_Type34',
    'Wilderness_Area3', 'Soil_Type12', 'Soil_Type2', 'Wilderness_Area1'
]

def predict():
    try:
        # Configuration MLflow
        tracking_uri = os.environ['NEON_DATABASE_URL'].replace('postgresql', 'postgresql+psycopg2')
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("forest_cover_type")
        logger.info("MLflow configuré")

        # Charger le modèle depuis S3
        s3 = boto3.client('s3')
        bucket = os.environ['S3_BUCKET']
        
        logging.info("Chargement du modèle depuis S3...")
        model_obj = s3.get_object(Bucket=bucket, Key='covertype/models/forest_cover_type_model.pkl')
        model = pickle.loads(model_obj['Body'].read())
        logger.info("Modèle chargé avec succès")

        # Charger les nouvelles données
        data_obj = s3.get_object(Bucket=bucket, Key='covertype/new_data/covtype_20.csv')
        data = pd.read_csv(data_obj['Body'])
        logger.info(f"Données chargées : {data.shape[0]} échantillons")

        # Store true labels
        true_labels = None
        if 'Cover_Type' in data.columns:
            true_labels = data['Cover_Type'].copy()
            logger.info("Vraies valeurs trouvées dans le dataset")
        
        # Select features and scale
        X = data[PREDICTION_COLUMNS].copy()
        scaler = MinMaxScaler(feature_range = (0,1))
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        X_scaled['Cover_Type'] = 0

        logger.info(f"Features normalisées avec MinMaxScaler: {X_scaled.shape[1]} colonnes")

# Make predictions with scaled data
        with mlflow.start_run():
            predictions = model.predict(X_scaled)
            data['Predicted_Cover_Type'] = predictions

            if true_labels is not None:
                accuracy = accuracy_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions, average='weighted')
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"F1 Score: {f1:.4f}")
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)

            # Save results
            output_key = "covertype/predictions/predictions_covtype_20.csv"
            csv_buffer = data.to_csv(index=False)
            s3.put_object(Bucket=bucket, Key=output_key, Body=csv_buffer)
            logger.info(f"Prédictions sauvegardées dans s3://{bucket}/{output_key}")

    except Exception as e:
        logger.error(f"Erreur : {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Démarrage du programme de prédiction...")
    predict()
    logger.info("Programme terminé")
import pytest
import boto3
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

test_results = []  # Pour stocker les résultats des tests

@pytest.fixture
def reference_data():
    s3 = boto3.client('s3')
    bucket = os.environ['S3_BUCKET']
    data_obj = s3.get_object(Bucket=bucket, Key='covertype/reference/covtype_80.csv')
    return pd.read_csv(data_obj['Body'])

@pytest.fixture
def new_data():
    s3 = boto3.client('s3')
    bucket = os.environ['S3_BUCKET']
    data_obj = s3.get_object(Bucket=bucket, Key='covertype/new_data/covtype_20.csv')
    return pd.read_csv(data_obj['Body'])

def save_test_results():
    """Sauvegarde les résultats dans un CSV sur S3"""
    df_results = pd.DataFrame(test_results)
    s3 = boto3.client('s3')
    bucket = os.environ['S3_BUCKET']
    
    # Sauvegarder en CSV
    csv_buffer = df_results.to_csv(index=False).encode()
    s3.put_object(
        Bucket=bucket,
        Key=f'test_reports/test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        Body=csv_buffer
    )

# Modifier les tests pour ajouter leurs résultats
def test_schema_consistency(reference_data, new_data):
    try:
        assert list(reference_data.columns) == list(new_data.columns)
        test_results.append({
            'test_name': 'schema_consistency',
            'status': 'PASSED',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except AssertionError:
        test_results.append({
            'test_name': 'schema_consistency',
            'status': 'FAILED',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        raise

# [Autres tests similaires...]

def pytest_sessionfinish(session):
    """Appelé après tous les tests"""
    save_test_results()


        

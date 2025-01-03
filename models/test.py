import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

@pytest.fixture
def reference_data():
    return pd.read_csv('data/reference.csv')

@pytest.fixture
def new_data():
    return pd.read_csv('data/new_data.csv')

def test_schema_consistency(reference_data, new_data):
    """Vérifie que les deux datasets ont les mêmes colonnes"""
    assert list(reference_data.columns) == list(new_data.columns)

def test_data_types(reference_data, new_data):
    """Vérifie que les types de données sont cohérents"""
    for col in reference_data.columns:
        assert reference_data[col].dtype == new_data[col].dtype

def test_value_ranges(reference_data, new_data):
    """Vérifie que les nouvelles données sont dans les plages acceptables"""
    for col in reference_data.columns:
        if pd.api.types.is_numeric_dtype(reference_data[col]):
            ref_min, ref_max = reference_data[col].min(), reference_data[col].max()
            new_min, new_max = new_data[col].min(), new_data[col].max()
            # Tolère une variation de 20% au-delà des limites de référence
            assert new_min >= ref_min * 0.8, f"Valeurs trop basses dans {col}"
            assert new_max <= ref_max * 1.2, f"Valeurs trop hautes dans {col}"

def test_missing_values(reference_data, new_data):
    """Vérifie le pourcentage de valeurs manquantes"""
    max_missing_pct = 0.1  # Maximum 10% de valeurs manquantes
    for col in new_data.columns:
        missing_pct = new_data[col].isnull().mean()
        assert missing_pct <= max_missing_pct, f"Trop de valeurs manquantes dans {col}"

def test_statistical_distribution(reference_data, new_data):
    """Compare les distributions statistiques de base"""
    numerical_columns = reference_data.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    
    for col in numerical_columns:
        ref_scaled = scaler.fit_transform(reference_data[col].values.reshape(-1, 1))
        new_scaled = scaler.transform(new_data[col].values.reshape(-1, 1))
        
        ref_mean = np.mean(ref_scaled)
        new_mean = np.mean(new_scaled)
        
        # Tolère une différence de moyenne de 0.5 écart-type
        assert abs(ref_mean - new_mean) < 0.5, f"Distribution trop différente pour {col}"


        

"""
Tests de l'API FastAPI.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import app
from features.pipeline import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def client():
    """TestClient avec lifespan (charge le modele au demarrage)."""
    with TestClient(app) as c:
        yield c


def make_sample_features() -> dict:
    """Genere des features synthetiques valides."""
    np.random.seed(42)
    return {col: float(np.random.normal(0, 1)) for col in FEATURE_COLUMNS}


# ============================================================
# TESTS HEALTH
# ============================================================

def test_health_returns_200(client):
    """Health check doit retourner 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_model_loaded(client):
    """Health doit indiquer modele charge."""
    response = client.get("/health")
    data = response.json()
    assert data['model_loaded'] is True
    assert data['status'] == 'healthy'


# ============================================================
# TESTS PREDICT
# ============================================================

def test_predict_valid_request(client):
    """Prediction valide doit retourner 200."""
    payload = {
        "features": make_sample_features(),
        "position": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_action_valid(client):
    """Action doit etre BUY, SELL ou HOLD."""
    payload = {
        "features": make_sample_features(),
        "position": 0,
    }
    response = client.post("/predict", json=payload)
    data = response.json()
    assert data['action'] in ['BUY', 'SELL', 'HOLD']


def test_predict_confidence_bounds(client):
    """Confiance doit etre entre 0 et 1."""
    payload = {
        "features": make_sample_features(),
        "position": 0,
    }
    response = client.post("/predict", json=payload)
    data = response.json()
    assert 0.0 <= data['confidence'] <= 1.0


def test_predict_has_timestamp(client):
    """Reponse doit avoir un timestamp."""
    payload = {
        "features": make_sample_features(),
        "position": 0,
    }
    response = client.post("/predict", json=payload)
    data = response.json()
    assert 'timestamp' in data


def test_predict_missing_features_handled(client):
    """Features manquantes doivent etre gerees sans crash."""
    payload = {
        "features": {"return_1": 0.001},  # Presque tout manquant
        "position": 0,
    }
    response = client.post("/predict", json=payload)
    # Doit retourner 200 (gestion gracieuse) ou 422 (validation)
    assert response.status_code in [200, 422]


def test_predict_invalid_position(client):
    """Position invalide doit retourner 422."""
    payload = {
        "features": make_sample_features(),
        "position": 5,  # Invalide
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ============================================================
# TESTS MODELS
# ============================================================

def test_models_available(client):
    """Endpoint modeles doit retourner une liste."""
    response = client.get("/models/available")
    assert response.status_code == 200
    data = response.json()
    assert 'available' in data
    assert 'current' in data


def test_metrics_endpoint(client):
    """Endpoint metriques doit retourner des donnees."""
    response = client.get("/metrics/latest")
    assert response.status_code == 200
    data = response.json()
    assert 'model_name' in data


def test_features_required(client):
    """Endpoint features doit lister toutes les features."""
    response = client.get("/features/required")
    assert response.status_code == 200
    data = response.json()
    assert data['count'] == len(FEATURE_COLUMNS)
    assert set(data['features']) == set(FEATURE_COLUMNS)


# ============================================================
# TEST DE CHARGE BASIQUE
# ============================================================

def test_load_10_requests(client):
    """10 requetes consecutives doivent toutes reussir."""
    payload = {
        "features": make_sample_features(),
        "position": 0,
    }
    for i in range(10):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, \
            f"Requete {i + 1} echouee : {response.status_code}"

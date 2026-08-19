from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


# ============================================================
# 1. TESTE DA HOME
# ============================================================

def test_home():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "online"


# ============================================================
# 2. TESTE DO HEALTH CHECK
# ============================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# ============================================================
# 3. TESTE DAS INFORMAÇÕES DO MODELO
# ============================================================

def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "XGBoost"
    assert data["calibration"] == "Platt"
    assert data["number_of_features"] == 75
    assert data["status"] == "ready"


# ============================================================
# 4. TESTE DE ERRO QUANDO FALTAM FEATURES
# ============================================================

def test_predict_missing_features():

    payload = {
        "features": {
            "AMT_CREDIT": 100000
        }
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 400

    data = response.json()

    assert "Missing required features" in data["detail"]
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from src.predict import predict_credit_risk, model_info


# ============================================================
# 1. CRIAR A APLICAÇÃO FASTAPI
# ============================================================

app = FastAPI(
    title="Credit Risk Decision System",
    description=(
        "Machine Learning API for credit risk assessment. "
        "The system estimates Probability of Default (PD), "
        "Expected Loss, Expected Profit and generates a "
        "credit decision."
    ),
    version="1.0.0"
)


# ============================================================
# 2. MODELO DE ENTRADA
# ============================================================

class CreditApplication(BaseModel):
    features: Dict[str, Any]


# ============================================================
# 3. HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Credit Risk Decision System API",
        "status": "online"
    }


# ============================================================
# 4. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 5. INFORMAÇÕES DO MODELO
# ============================================================

@app.get("/model-info")
def get_model_info():

    try:
        return model_info()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# 6. PREDIÇÃO DE RISCO DE CRÉDITO
# ============================================================

@app.post("/predict")
def predict(application: CreditApplication):

    try:

        result = predict_credit_risk(
            application.features
        )

        # Converter possíveis numpy types para tipos Python
        clean_result = {}

        for key, value in result.items():

            if hasattr(value, "item"):
                value = value.item()

            clean_result[key] = value

        return clean_result

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
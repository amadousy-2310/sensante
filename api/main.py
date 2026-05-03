# api/main.py
# API FastAPI pour SenSante - Assistant pré-diagnostic médical

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# --- Charger le modèle et les encodeurs au démarrage ---

print("Chargement du modèle...")

model       = joblib.load("models/model.pkl")
le_sexe     = joblib.load("models/encoder_sexe.pkl")
le_region   = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print(f"Modèle chargé : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")
# Créer l'application
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

# Route de base : vérifier que l'API fonctionne
@app.get("/health")
def health_check():
    """Vérification de l'état de l'API."""
    return {
        "status": "ok",
        "message": "SenSante API is running"
    }

# --- Schémas Pydantic ---

class PatientInput(BaseModel):
    """Données d'entrée : les symptômes d'un patient."""

    age: int = Field(..., ge=0, le=120, description="Âge en années")
    sexe: str = Field(..., description="Sexe : M ou F")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Température en Celsius")
    tension_sys: int = Field(..., ge=60, le=250, description="Tension systolique")
    toux: bool = Field(..., description="Présence de toux")
    fatigue: bool = Field(..., description="Présence de fatigue")
    maux_tete: bool = Field(..., description="Présence de maux de tête")
    region: str = Field(..., description="Région du Sénégal")


class DiagnosticOutput(BaseModel):
    """Données de sortie : le résultat du diagnostic."""

    diagnostic: str = Field(..., description="Diagnostic prédit")
    probabilite: float = Field(..., description="Probabilité du diagnostic")
    confiance: str = Field(..., description="Niveau de confiance")
    message: str = Field(..., description="Recommandation")
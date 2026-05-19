# api/main.py
# API FastAPI pour SenSante - Assistant pré-diagnostic médical
# Lab 3 - Integration de Modeles IA - ESP/UCAD


from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# lab 5 - integration de Groq
import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement
load_dotenv()

# Client Groq (charge au demarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. /explain sera desactive.")

# --- 1. Schémas Pydantic  ---

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

class ExplainInput(BaseModel):
    """Données d'entrée pour l'explication LLM."""
    diagnostic: str
    probabilite: float
    age: int
    sexe: str
    temperature: float
    region: str


class ExplainOutput(BaseModel):
    """Données de sortie pour l'explication LLM."""
    explication: str
    modele_llm: str = "llama-3.1-8b-instant"

# --- 2. Application FastAPI ---

app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)
# Autoriser les requetes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En dev : tout accepter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Chargement du modèle ---

print("Chargement du modèle...")
model        = joblib.load("models/model.pkl")
le_sexe      = joblib.load("models/encoder_sexe.pkl")
le_region    = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modèle chargé : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")


# --- 4. Routes ---

@app.get("/health")
def health_check():
    """Vérification de l'état de l'API."""
    return {
        "status": "ok",
        "message": "SenSante API is running"
    }


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    """
    Predire un diagnostic a partir des symptomes d'un patient.
    Recoit les symptomes en JSON, renvoie le diagnostic,
    la probabilite et une recommandation.
    """

    # 1. Encoder les variables categoriques
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F."
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # 2. Construire le vecteur de features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # 3. Predire
    diagnostic = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    proba_max = float(probas.max())

    # 4. Determiner le niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # 5. Generer la recommandation
    messages = {
        "palu":   "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typh":   "Suspicion de typhoide. Consultation medicale necessaire.",
        "sain":   "Pas de pathologie detectee. Continuez a surveiller."
    }

    # 6. Renvoyer le resultat
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )
# Exercice 1 — Endpoint GET /model-info
@app.get("/model-info")
def model_info():
    """Informations sur le modèle chargé."""
    return {
        "type": type(model).__name__,
        "nombre_arbres": model.n_estimators,
        "classes": list(model.classes_),
        "nombre_features": model.n_features_in_
    }
# SYSTEM PROMPT pour le LLM
SYSTEM_PROMPT = """Tu es un assistant medical senegalais. Tu recois un diagnostic et des donnees patient. Explique le resultat en francais simple, comme un medecin parlerait a son patient. Sois rassurant mais recommande toujours une consultation medicale. Maximum 3 phrases. Ne fais JAMAIS de diagnostic toi-meme. Tu expliques uniquement le diagnostic fourni."""


@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en francais avec un LLM."""
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Cle API non configuree.",
            modele_llm="aucun"
        )
    
    # Construire le user prompt
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, "
        f"region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"
    
    return ExplainOutput(explication=explication, modele_llm="llama-3.1-8b-instant")
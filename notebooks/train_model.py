# notebooks/train_model.py
# LAB 2 - Étape par étape selon le document

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

print("=" * 60)
print("LAB 2 - Entraînement et sérialisation du modèle SénSanté")
print("=" * 60)

# ============================================================
# Étape 2 : Charger et préparer les données
# ============================================================
print("\n" + "=" * 60)
print("Étape 2 : Charger et préparer les données")
print("=" * 60)

# Étape 2.1 : Charger le dataset
print("\n Étape 2.1 : Chargement du dataset...")
df = pd.read_csv("data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# Étape 2.2 : Préparer les features et la cible
print("\n Étape 2.2 : Encodage des variables catégoriques...")
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys', 
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")  # (500, 8)
print(f"Cible : {y.shape}")     # (500,)

# ============================================================
# Étape 3 : Séparer entraînement et test
# ============================================================
print("\n" + "=" * 60)
print("Étape 3 : Séparer entraînement et test")
print("=" * 60)

print("\n Étape 3.1 : Séparation des données...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

# ============================================================
# Étape 4 : Entraîner le modèle
# ============================================================
print("\n" + "=" * 60)
print("Étape 4 : Entraîner le modèle")
print("=" * 60)

print("\n Étape 4.1 : Création et entraînement du RandomForest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Modèle entraîné !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# ============================================================
# Étape 5 : Évaluer le modèle
# ============================================================
print("\n" + "=" * 60)
print("Étape 5 : Évaluer le modèle")
print("=" * 60)

# Étape 5.1 : Prédire sur les données de test
print("\n Étape 5.1 : Prédiction sur les données de test...")
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prédiction': y_pred[:10]
})
print("\nComparaison des 10 premiers patients :")
print(comparison)

# Étape 5.2 : Calculer l'accuracy
print("\n Étape 5.2 : Calcul de l'accuracy...")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

# Étape 5.3 : Matrice de confusion et rapport
print("\n Étape 5.3 : Matrice de confusion et rapport de classification...")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Étape 5.4 : Visualiser la matrice de confusion (optionnel)
print("\n Étape 5.4 : Visualisation de la matrice de confusion...")
os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
plt.xlabel('Prédiction du modèle')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SénSanté')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
print("Figure sauvegardée dans figures/confusion_matrix.png")
plt.close()

# ============================================================
# Étape 6 : Sérialiser le modèle
# ============================================================
print("\n" + "=" * 60)
print("Étape 6 : Sérialiser le modèle")
print("=" * 60)

# Étape 6.1 : Sauvegarder le modèle
print("\n Étape 6.1 : Sauvegarde du modèle...")
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")

size = os.path.getsize("models/model.pkl")
print(f"Modèle sauvegardé : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# Étape 6.2 : Sauvegarder aussi les encodeurs
print("\n Étape 6.2 : Sauvegarde des encodeurs et métadonnées...")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardés.")

# ============================================================
# Étape 7 : Tester le modèle sérialisé
# ============================================================
print("\n" + "=" * 60)
print("Étape 7 : Tester le modèle sérialisé")
print("=" * 60)

# Étape 7.1 : Recharger le modèle depuis le fichier
print("\n Étape 7.1 : Rechargement du modèle...")
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Étape 7.2 : Prédire pour un nouveau patient
print("\n Étape 7.2 : Prédiction pour un nouveau patient...")
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]

print(f"\nPatient : {nouveau_patient}")
print(f"Diagnostic prédit : {diagnostic}")
print("Probabilités :")
for classe, prob in zip(model_loaded.classes_, probas):
    print(f"  - {classe}: {prob:.1%}")
# ============================================================
# Exercices
# ============================================================
print("\n" + "=" * 60)
print("EXERCICES")
print("=" * 60)

# Exercice 1 — Importance des features
print("\nExercice 1 — Importance des features :")
print("-" * 40)

importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f"{name:20s} : {imp:.3f}")

# Exercice 2 — Tester avec d'autres patients
print("\nExercice 2 — Tester avec d'autres patients :")
print("-" * 40)

# 3 patients fictifs
patients_test = [
    {'nom': 'Jeune sans symptômes', 'age': 22, 'sexe': 'M', 'temperature': 36.5,
     'tension_sys': 110, 'toux': False, 'fatigue': False, 'maux_tete': False, 'region': 'Dakar'},
    {'nom': 'Adulte avec forte fièvre', 'age': 35, 'sexe': 'F', 'temperature': 40.0,
     'tension_sys': 115, 'toux': True, 'fatigue': True, 'maux_tete': True, 'region': 'Dakar'},
    {'nom': 'Patient âgé avec toux', 'age': 70, 'sexe': 'M', 'temperature': 38.0,
     'tension_sys': 130, 'toux': True, 'fatigue': True, 'maux_tete': False, 'region': 'Saint-Louis'}
]

for patient in patients_test:
    sexe_enc = le_sexe_loaded.transform([patient['sexe']])[0]
    region_enc = le_region_loaded.transform([patient['region']])[0]
    
    features = [
        patient['age'], sexe_enc, patient['temperature'],
        patient['tension_sys'], int(patient['toux']),
        int(patient['fatigue']), int(patient['maux_tete']), region_enc
    ]
    
    diag = model_loaded.predict([features])[0]
    print(f"{patient['nom']} : {diag}")

# Exercice 3 — Réflexion
print("\nExercice 3 — Réflexion :")
print("-" * 40)
print("""
Réponse : 89% d'accuracy n'est pas suffisant en contexte médical réel.
Les risques d'un faux diagnostic sont :
1. Faux négatif : patient malade considéré sain → absence de traitement, aggravation, contagion
2. Faux positif : patient sain considéré malade → traitements inutiles, stress, effets secondaires
En médecine, on privilégie le rappel (éviter les faux négatifs) plutôt que l'accuracy.
""")
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Fonction pour charger le modèle (cache pour éviter de le recharger à chaque interaction)
@st.cache_resource(show_spinner=False)
def load_model():
    # Récupérer le répertoire courant du script (app)
    base_dir = os.path.dirname(os.path.realpath(__file__))
    # Remonter d'un niveau pour atteindre le dossier racine
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))
    # Construire le chemin vers le modèle dans le dossier 'models'
    model_path = os.path.join(root_dir, "models", "catboost_model.pkl")
    
    if not os.path.exists(model_path):
        st.error(f"Le fichier modèle n'a pas été trouvé : {model_path}. Veuillez vérifier que le modèle est sauvegardé.")
        return None
    
    model = joblib.load(model_path)
    return model

# Charger le modèle
model = load_model()
if model is None:
    st.stop()  # Arrête l'exécution si le modèle n'est pas trouvé

# Titre de l'application
st.title("Prédiction du Risque de Défaut de Prêt")
st.write("Entrez les informations du client pour prédire le risque de défaut.")

# Création du formulaire d'input
credit_lines = st.number_input("Nombre de lignes de crédit en cours", min_value=0, value=2)
fico_score = st.number_input("Score FICO", min_value=300, max_value=850, value=650)
years_employed = st.number_input("Nombre d'années d'emploi", min_value=0, value=5)
loan_amt = st.number_input("Montant du prêt restant (en €)", min_value=0.0, value=5000.0)
income = st.number_input("Revenu annuel (en €)", min_value=0.0, value=50000.0)

# Calcul des transformations logarithmiques
log_loan_amt = np.log1p(loan_amt)
log_income = np.log1p(income)

# Préparer le DataFrame d'input pour le modèle, sans transformation logarithmique
input_data = pd.DataFrame({
    "credit_lines_outstanding": [credit_lines],
    "fico_score": [fico_score],
    "years_employed": [years_employed],
    "loan_amt_outstanding": [loan_amt],
    "income": [income]
})


st.write("Voici les données d'entrée préparées:")
st.dataframe(input_data)

# Bouton pour lancer la prédiction
if st.button("Prédire le risque de défaut"):
    # Obtenir la prédiction (0 = solvable, 1 = défaut)
    prediction = model.predict(input_data)
    # Obtenir la probabilité de la classe 1 (défaut)
    proba = model.predict_proba(input_data)[:, 1]
    
    # Affichage des résultats
    st.subheader("Résultats de la prédiction")
    st.write("Prédiction (0 = solvable, 1 = défaut):", int(prediction[0]))
    st.write("Probabilité de défaut: {:.2f}%".format(proba[0] * 100))
    
    # Message personnalisé
    if prediction[0] == 1:
        st.error("Attention ! Ce client présente un risque élevé de défaut.")
    else:
        st.success("Ce client semble avoir un risque faible de défaut.")

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate, KFold
import joblib
import os

# =========================
# Global Feature Sets
# =========================
feature_sets = {
    "base": ["credit_lines_outstanding", "fico_score", "years_employed"],
    "plus_loan": ["credit_lines_outstanding", "fico_score", "years_employed", "loan_amt_outstanding"],
    "plus_log_loan": ["credit_lines_outstanding", "fico_score", "years_employed", "log_loan_amt_outstanding"],
    "plus_log_loan_income": ["credit_lines_outstanding", "fico_score", "years_employed", "log_loan_amt_outstanding", "log_income"],
    "all": ["credit_lines_outstanding", "fico_score", "years_employed", "loan_amt_outstanding", "income"],
    "all_with_log": ["credit_lines_outstanding", "fico_score", "years_employed", "log_loan_amt_outstanding", "log_income"]
}

# =========================
# 1. Chargement & Prétraitement
# =========================
def load_and_preprocess_data(data_path: str, features: list, target: str, scale: bool = False, log_transform: bool = False):
    """
    Charge le dataset et prépare les données.
    
    Parameters:
        data_path (str): chemin vers le dataset.
        features (list): liste des colonnes à utiliser.
        target (str): nom de la cible.
        scale (bool): si True, applique un StandardScaler.
        log_transform (bool): si True, applique une transformation logarithmique sur 'loan_amt_outstanding' et 'income'.
        
    Returns:
        X: données d'entrée (potentiellement transformées et/ou standardisées).
        y: variable cible.
    """
    data = pd.read_csv(data_path)
    
    # Transformation logarithmique si demandée
    if log_transform:
        if 'loan_amt_outstanding' in data.columns:
            data['log_loan_amt_outstanding'] = np.log1p(data['loan_amt_outstanding'])
        if 'income' in data.columns:
            data['log_income'] = np.log1p(data['income'])
    
    X = data[features]
    y = data[target]
    
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y

# =========================
# 2. Régression Logistique avec Validation Croisée
# =========================
def run_logistic_regression_experiment(data_path: str, feature_set: str, weighted: bool, cv_folds: int = 5, random_state: int = 42):
    """
    Entraîne une régression logistique avec validation croisée sur le dataset en utilisant l'ensemble de features défini par feature_set.
    Calcule accuracy, precision, recall, f1, et AUC, et loggue l'expérience dans MLflow.
    
    Returns:
        metrics_mean: dictionnaire des scores moyens obtenus.
    """
    # Récupérer l'ensemble de features depuis le dictionnaire global
    features = feature_sets[feature_set]
    target = "default"
    # Appliquer transformation log si le nom de l'ensemble contient "log"
    log_transform = "log" in feature_set
    
    # Pour la régression logistique, le scaling est nécessaire
    X, y = load_and_preprocess_data(data_path, features, target, scale=True, log_transform=log_transform)
    
    class_weight = "balanced" if weighted else None
    model = LogisticRegression(class_weight=class_weight, random_state=random_state, max_iter=1000)
    
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'auc': 'roc_auc'
    }
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    metrics_mean = {metric: np.mean(results[f'test_{metric}']) for metric in scoring}
    
    run_name = f"LR_{'weighted' if weighted else 'standard'}_{feature_set}"
    mlflow.set_experiment("logistic_regression_experiments")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("weighted", weighted)
        mlflow.log_param("features", features)
        mlflow.log_param("cv_folds", cv_folds)
        for metric, value in metrics_mean.items():
            mlflow.log_metric(metric, value)
    
    print(f"Run '{run_name}' - Metrics moyennes (5-fold):")
    for metric, value in metrics_mean.items():
        print(f"  {metric}: {value:.4f}")
    print("")
    
    return metrics_mean

# =========================
# 3. Random Forest avec Validation Croisée
# =========================
def run_random_forest_experiment(data_path: str, feature_set: str = 'base', target: str = "default", 
                                 cv_folds: int = 5, random_state: int = 42, n_estimators: int = 100):
    """
    Entraîne un modèle RandomForest avec validation croisée sur le dataset et loggue l'expérience dans MLflow.
    Calcule accuracy, precision, recall, f1, et AUC.
    
    Returns:
        metrics_mean: dictionnaire des scores moyens obtenus.
    """
    features = feature_sets[feature_set]
    target = "default"
    log_transform = "log" in feature_set
    
    # Pour Random Forest, scaling n'est pas indispensable
    X, y = load_and_preprocess_data(data_path, features, target, scale=False, log_transform=log_transform)
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'auc': 'roc_auc'
    }
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    metrics_mean = {metric: np.mean(results[f'test_{metric}']) for metric in scoring}
    
    run_name = f"RF_n{n_estimators}_{feature_set}"
    mlflow.set_experiment("random_forest_experiments")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("features", features)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("cv_folds", cv_folds)
        for metric, value in metrics_mean.items():
            mlflow.log_metric(metric, value)
    
    print(f"Run '{run_name}' - Metrics moyennes (5-fold):")
    for metric, value in metrics_mean.items():
        print(f"  {metric}: {value:.4f}")
    print("")
    
    return metrics_mean

# =========================
# 4. CatBoost avec Validation Croisée
# =========================
def run_catboost_experiment(data_path: str, feature_set: str = 'base', target: str = "default", 
                            cv_folds: int = 5, random_state: int = 42, iterations: int = 100):
    """
    Entraîne un modèle CatBoostClassifier avec validation croisée sur le dataset et loggue l'expérience dans MLflow.
    Calcule accuracy, precision, recall, f1, et AUC.
    
    Returns:
        metrics_mean: dictionnaire des scores moyens obtenus.
    """
    features = feature_sets[feature_set]
    target = "default"
    log_transform = "log" in feature_set
    
    # CatBoost gère bien les données brutes, scaling non nécessaire
    X, y = load_and_preprocess_data(data_path, features, target, scale=False, log_transform=log_transform)
    
    model = CatBoostClassifier(iterations=iterations, random_state=random_state, verbose=0)
    
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'auc': 'roc_auc'
    }
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    metrics_mean = {metric: np.mean(results[f'test_{metric}']) for metric in scoring}
    
    run_name = f"CatBoost_{iterations}_{feature_set}"
    mlflow.set_experiment("catboost_experiments")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "CatBoostClassifier")
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("features", features)
        mlflow.log_param("iterations", iterations)
        mlflow.log_param("cv_folds", cv_folds)
        for metric, value in metrics_mean.items():
            mlflow.log_metric(metric, value)
    
    print(f"Run '{run_name}' - Metrics moyennes (5-fold):")
    for metric, value in metrics_mean.items():
        print(f"  {metric}: {value:.4f}")
    print("")
    
    return metrics_mean

def train_and_save_final_catboost(data_path: str, feature_set: str = 'base', target: str = "default", 
                                  random_state: int = 42, iterations: int = 100):
    """
    Entraîne le modèle CatBoostClassifier sur l'ensemble du dataset sans log transformation ni normalisation,
    et sauvegarde le modèle final dans 'models/catboost_model.pkl'.
    
    Parameters:
        data_path (str): chemin vers le dataset.
        feature_set (str): ensemble de features à utiliser (ex: 'base', 'plus_loan', 'all', etc.).
        target (str): nom de la cible (par défaut "default").
        random_state (int): graine pour la reproductibilité.
        iterations (int): nombre d'itérations pour CatBoost.
        
    Returns:
        model: Le modèle CatBoost entraîné sur l'ensemble des données.
    """
    # Récupérer l'ensemble de features à partir du dictionnaire global 'feature_sets'
    features = feature_sets[feature_set]
    
    # Charger les données sans appliquer de scaling ni de log transformation
    # Ici, on force log_transform=False et scale=False pour une version "normale"
    X, y = load_and_preprocess_data(data_path, features, target, scale=False, log_transform=False)
    
    # Instancier et entraîner le modèle CatBoost
    model = CatBoostClassifier(iterations=iterations, random_state=random_state, verbose=0)
    model.fit(X, y)
    
    # Créer le dossier models s'il n'existe pas et sauvegarder le modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/catboost_model.pkl")
    print("Modèle CatBoost final sauvegardé dans models/catboost_model.pkl")
    
    return model


# =========================
# Main: Appel de toutes les expériences
# =========================
if __name__ == "__main__":
    # =========================
    # Saving du modèle en pkl pour l'application
    # =========================
    
    data_path = "data/Loan_Data.csv"
    print("=== Entraînement final du modèle CatBoost ===")
    final_model = train_and_save_final_catboost(data_path, feature_set="base")

    # =========================
    # ML Flow model testing
    # =========================

    #print("=== Régression Logistique ===")
    # for fs in feature_sets.keys():
    #     # On exécute pour chaque ensemble de features, en testant à la fois la version standard et pondérée
    #     print(f"Feature set: {fs}")
    #     run_logistic_regression_experiment(data_path, feature_set=fs, weighted=False)
    #     run_logistic_regression_experiment(data_path, feature_set=fs, weighted=True)
    
    # print("=== Random Forest ===")
    # for fs in feature_sets.keys():
    #     run_random_forest_experiment(data_path, feature_set=fs)
    
    # print("=== CatBoost ===")
    # for fs in feature_sets.keys():
    #     run_catboost_experiment(data_path, feature_set=fs)
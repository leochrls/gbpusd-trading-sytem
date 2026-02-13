"""
Definition des modeles ML avec hyperparametres optimises
pour RTX Ada 1000 (legers, rapides, efficaces).
"""

import pickle
from pathlib import Path
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_models() -> Dict[str, Any]:
    """
    Retourne tous les modeles avec hyperparametres.
    Tous optimises pour etre legers (CPU/GPU limite).

    Returns:
        Dict nom -> instance modele
    """
    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            random_state=42,
            n_jobs=-1,
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }
    logger.info(f"Modeles initialises : {list(models.keys())}")
    return models


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "val",
) -> Dict[str, float]:
    """
    Evalue un modele sur un split.

    Args:
        model: Modele sklearn/lgbm fitte
        X: Features
        y: Target
        split_name: Nom du split pour les logs

    Returns:
        Dict de metriques statistiques
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, y_proba)),
        'n_samples': len(y),
        'split': split_name,
    }

    logger.info(f"\n{split_name.upper()} - {model.__class__.__name__}")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

    return metrics


def get_feature_importance(
    model: Any,
    feature_names: list,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extrait et trie les feature importances.

    Args:
        model: Modele fitte
        feature_names: Noms des features
        top_n: Nombre de features a afficher

    Returns:
        DataFrame trie par importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = abs(model.coef_[0])
    else:
        logger.warning(f"Pas de feature_importances pour {type(model).__name__}")
        return pd.DataFrame()

    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False).head(top_n)

    return df_imp


def save_model(model: Any, name: str, version: str = 'v1') -> str:
    """Sauvegarde un modele sklearn/lgbm.

    Args:
        model: Modele fitte a sauvegarder.
        name: Nom du modele (ex: 'lightgbm', 'best').
        version: Version du modele.

    Returns:
        Chemin du fichier sauvegarde.
    """
    path = Path(f'models/{version}/ml_{name}.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    logger.success(f"Modele sauvegarde : {path}")
    return str(path)


def load_model(name: str, version: str = 'v1') -> Any:
    """Charge un modele sauvegarde.

    Args:
        name: Nom du modele.
        version: Version du modele.

    Returns:
        Instance du modele charge.
    """
    path = Path(f'models/{version}/ml_{name}.pkl')

    with open(path, 'rb') as f:
        model = pickle.load(f)

    logger.success(f"Modele charge : {path}")
    return model

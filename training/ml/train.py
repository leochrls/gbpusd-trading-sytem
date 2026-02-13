"""
Pipeline d'entrainement ML complet.
Train sur 2022, validation sur 2023, test final sur 2024.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve

from evaluation.backtester import Backtester
from features.pipeline import FEATURE_COLUMNS
from training.baseline.strategies import Action, BaseStrategy
from training.ml.models import (
    evaluate_model,
    get_feature_importance,
    get_models,
    save_model,
)
from training.ml.prepare_data import load_ml_splits


class MLStrategy(BaseStrategy):
    """
    Strategie basee sur predictions ML.
    Wrapping d'un modele sklearn pour le backtester.

    Attributes:
        model: Modele sklearn/lgbm fitte.
        threshold: Seuil de probabilite pour BUY/SELL.
    """

    def __init__(self, model, name: str, threshold: float = 0.5) -> None:
        super().__init__(name=f"ML_{name}")
        self.model = model
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Genere signaux depuis predictions ML.

        Args:
            df: DataFrame M15 avec features.

        Returns:
            Series de signaux Action.
        """
        X = df[FEATURE_COLUMNS].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]

        signals = pd.Series(Action.HOLD, index=df.index, name='signal')
        signals[proba >= self.threshold] = Action.BUY
        signals[proba < (1 - self.threshold)] = Action.SELL

        return signals


def _safe_write_image(fig: go.Figure, path: str, **kwargs) -> None:
    """Write image avec fallback si kaleido n'est pas installe."""
    try:
        fig.write_image(path, **kwargs)
    except (ImportError, ValueError) as e:
        logger.warning(f"write_image impossible ({e}). Installez kaleido: pip install kaleido")


def plot_roc_curves(
    models_results: Dict,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    save_path: str = 'evaluation/ml_roc_curves.html',
) -> None:
    """Plot ROC curves pour tous les modeles.

    Args:
        models_results: Dict nom -> resultats (contenant 'model').
        X_val: Features validation.
        y_val: Target validation.
        save_path: Chemin de sauvegarde HTML.
    """
    fig = go.Figure()

    colors = ['#00ff88', '#ff4444', '#4488ff', '#ffaa00']

    for i, (name, result) in enumerate(models_results.items()):
        model = result['model']
        y_proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={roc_auc:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    # Diagonale random
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random (AUC=0.5)',
        line=dict(color='white', width=1, dash='dash'),
    ))

    fig.update_layout(
        template='plotly_dark',
        title='Courbes ROC - Comparaison Modeles',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path)
    #_safe_write_image(fig, save_path.replace('.html', '.png'), width=1000, height=600)
    logger.success(f"ROC curves sauvegardees : {save_path}")


def plot_feature_importances(
    models_results: Dict,
    save_path: str = 'evaluation/ml_feature_importances.html',
) -> None:
    """Plot feature importances pour tous les modeles.

    Args:
        models_results: Dict nom -> resultats (contenant 'feature_importance').
        save_path: Chemin de sauvegarde HTML.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(models_results.keys()),
        vertical_spacing=0.2,
        horizontal_spacing=0.15,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for i, (name, result) in enumerate(models_results.items()):
        df_imp = result.get('feature_importance', pd.DataFrame())
        if df_imp.empty:
            continue

        row, col = positions[i]
        fig.add_trace(
            go.Bar(
                x=df_imp['importance'],
                y=df_imp['feature'],
                orientation='h',
                name=name,
                marker_color='#4488ff',
            ),
            row=row, col=col,
        )

    fig.update_layout(
        template='plotly_dark',
        title='Feature Importances par Modele',
        height=900,
        showlegend=False,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path)
    #(fig, save_path.replace('.html', '.png'), width=1400, height=900)
    logger.success(f"Feature importances sauvegardees : {save_path}")


def main() -> Tuple[Dict, object]:
    """Pipeline d'entrainement ML complet.

    Returns:
        Tuple (models_results, best_model)
    """
    logger.info("=== DEBUT ENTRAINEMENT ML ===")

    # Chargement donnees
    X_train, y_train, X_val, y_val, X_test, y_test = load_ml_splits(
        FEATURE_COLUMNS
    )

    # Modeles
    models = get_models()
    models_results: Dict = {}
    all_metrics: Dict = {}

    # === ENTRAINEMENT ET VALIDATION ===
    logger.info("\n PHASE 1 : Entrainement sur 2022 + Validation sur 2023")

    for name, model in models.items():
        logger.info(f"\n Entrainement {name}...")

        # Train sur 2022
        model.fit(X_train, y_train)
        logger.success(f"{name} entraine")

        # Evaluation train (verif overfitting)
        train_metrics = evaluate_model(model, X_train, y_train, "train")

        # Evaluation validation 2023
        val_metrics = evaluate_model(model, X_val, y_val, "val")

        # Feature importances
        df_imp = get_feature_importance(model, FEATURE_COLUMNS)

        # Sauvegarde modele
        save_model(model, name)

        models_results[name] = {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': df_imp,
        }

        all_metrics[name] = {
            'train': train_metrics,
            'val': val_metrics,
        }

        # Detection overfitting
        overfit_gap = train_metrics['accuracy'] - val_metrics['accuracy']
        if overfit_gap > 0.05:
            logger.warning(
                f"Possible overfitting {name} : "
                f"train={train_metrics['accuracy']:.3f} "
                f"val={val_metrics['accuracy']:.3f} "
                f"gap={overfit_gap:.3f}"
            )

    # === SELECTION MEILLEUR MODELE (sur val 2023) ===
    best_name = max(
        models_results.keys(),
        key=lambda k: models_results[k]['val_metrics']['roc_auc'],
    )
    best_model = models_results[best_name]['model']

    logger.success(
        f"\n Meilleur modele (val ROC-AUC) : {best_name} "
        f"({models_results[best_name]['val_metrics']['roc_auc']:.4f})"
    )

    # Sauvegarde meilleur modele separement
    save_model(best_model, 'best')

    # === VISUALISATIONS ===
    logger.info("\n Generation des visualisations...")
    plot_roc_curves(models_results, X_val, y_val)
    plot_feature_importances(models_results)

    # === BACKTESTING FINANCIER ===
    logger.info("\n PHASE 2 : Evaluation financiere (backtesting)")

    # Reconstruction DataFrames avec features pour backtester
    df_val_full = pd.read_parquet('data/splits/val_features.parquet')
    df_test_full = pd.read_parquet('data/splits/test_features.parquet')

    # Aligne avec X_val (meme index apres dropna)
    df_val_bt = df_val_full.loc[X_val.index]
    df_test_bt = df_test_full.loc[X_test.index]

    if 'timestamp' in df_val_bt.columns:
        df_val_bt = df_val_bt.set_index('timestamp')
        df_test_bt = df_test_bt.set_index('timestamp')

    backtester = Backtester(
        initial_capital=10000,
        transaction_cost=0.0002,
        slippage=0.0001,
    )

    financial_results: Dict = {}

    for name, result in models_results.items():
        strategy = MLStrategy(result['model'], name)

        # Validation 2023
        val_fin = backtester.run(strategy, df_val_bt, split_name=f"val_{name}")
        financial_results[name] = {'val': val_fin.to_dict()}

    # Equity curves validation
    backtester.plot_equity_curve(
        save_path='evaluation/ml_equity_val.html'
    )

    # Reset pour test
    backtester._results = {}
    financial_test: Dict = {}

    for name, result in models_results.items():
        strategy = MLStrategy(result['model'], name)
        test_fin = backtester.run(strategy, df_test_bt, split_name=f"test_{name}")
        financial_test[name] = {'test': test_fin.to_dict()}

    # Equity curves test
    backtester.plot_equity_curve(
        save_path='evaluation/ml_equity_test.html'
    )

    # === TEST FINAL 2024 - MEILLEUR MODELE UNIQUEMENT ===
    logger.info(f"\n TEST FINAL 2024 - {best_name}")
    best_test_metrics = evaluate_model(best_model, X_test, y_test, "TEST_FINAL_2024")

    all_metrics[best_name]['test_final'] = best_test_metrics

    # === SAUVEGARDE RESULTATS ===
    results_summary = {
        'best_model': best_name,
        'best_val_roc_auc': models_results[best_name]['val_metrics']['roc_auc'],
        'best_test_final': best_test_metrics,
        'all_models': {
            name: {
                'train': r['train_metrics'],
                'val': r['val_metrics'],
            }
            for name, r in models_results.items()
        },
        'financial_val': {
            name: v['val']
            for name, v in financial_results.items()
        },
        'financial_test': {
            name: v['test']
            for name, v in financial_test.items()
        },
    }

    Path('evaluation').mkdir(parents=True, exist_ok=True)
    with open('evaluation/ml_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    logger.success("Resultats sauvegardes : evaluation/ml_results.json")

    # === RESUME FINAL ===
    logger.info("\n" + "=" * 60)
    logger.info("RESUME ML")
    logger.info("=" * 60)
    for name, r in models_results.items():
        v = r['val_metrics']
        marker = ">>>" if name == best_name else "   "
        logger.info(
            f"{marker} {name:25s} | "
            f"Val ROC-AUC: {v['roc_auc']:.4f} | "
            f"Val F1: {v['f1']:.4f} | "
            f"Val Acc: {v['accuracy']:.4f}"
        )
    logger.info("=" * 60)
    logger.success("=== FIN ENTRAINEMENT ML ===")

    return models_results, best_model


if __name__ == "__main__":
    main()

"""
Registry de modeles : charge et cache les modeles en memoire.
Thread-safe, supporte ML (pickle) et RL (PyTorch).
"""
import json
import pickle
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


@dataclass
class ModelInfo:
    """Metadonnees d'un modele charge."""

    name: str
    version: str
    type: str          # 'ml' ou 'rl'
    path: str
    loaded_at: str
    metrics: Dict[str, Any]


class ModelRegistry:
    """
    Registry thread-safe pour charger et cacher les modeles.

    - Charge automatiquement le meilleur modele disponible
    - Supporte ML (sklearn/lgbm) et RL (PyTorch DQN)
    - Cache en memoire (pas de rechargement a chaque requete)
    - Thread-safe avec verrou

    Usage:
        registry = ModelRegistry()
        registry.load_best_model()
        model, info = registry.get_model()
    """

    def __init__(self, models_dir: str = 'models') -> None:
        self.models_dir = Path(models_dir)
        self._model = None
        self._model_info: Optional[ModelInfo] = None
        self._pipeline = None
        self._lock = threading.Lock()

    def _load_feature_pipeline(self):
        """Charge le pipeline de features sauvegarde."""
        pipeline_path = self.models_dir / 'v1' / 'feature_pipeline.pkl'

        if not pipeline_path.exists():
            logger.warning(f"Pipeline features non trouve : {pipeline_path}")
            return None

        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)

        logger.success(f"Feature pipeline charge : {pipeline_path}")
        return pipeline

    def _load_ml_model(self, version: str = 'v1'):
        """Charge le meilleur modele ML."""
        path = self.models_dir / version / 'ml_best.pkl'

        if not path.exists():
            # Fallback sur lightgbm
            path = self.models_dir / version / 'ml_lightgbm.pkl'

        if not path.exists():
            raise FileNotFoundError(
                f"Aucun modele ML trouve dans {self.models_dir / version}"
            )

        with open(path, 'rb') as f:
            model = pickle.load(f)

        # Charge metriques si dispo
        metrics: Dict[str, Any] = {}
        metrics_path = Path('evaluation/ml_results.json')
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                ml_data = json.load(f)
                metrics = ml_data.get('best_test_final', {})

        info = ModelInfo(
            name=type(model).__name__,
            version=version,
            type='ml',
            path=str(path),
            loaded_at=datetime.now().isoformat(),
            metrics=metrics,
        )

        logger.success(f"Modele ML charge : {path}")
        return model, info

    def _load_rl_model(self, version: str = 'v1'):
        """Charge le meilleur agent RL."""
        from training.rl.agent import DQNAgent

        path = self.models_dir / version / 'rl_best.pth'

        if not path.exists():
            raise FileNotFoundError(f"Modele RL non trouve : {path}")

        agent = DQNAgent.load(str(path))
        agent.epsilon = 0.0  # Pas d'exploration en prod

        # Charge metriques si dispo
        metrics: Dict[str, Any] = {}
        metrics_path = Path('evaluation/rl_test_results.json')
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

        info = ModelInfo(
            name='DQN',
            version=version,
            type='rl',
            path=str(path),
            loaded_at=datetime.now().isoformat(),
            metrics=metrics,
        )

        logger.success(f"Modele RL charge : {path}")
        return agent, info

    def load_best_model(self, version: str = 'v1') -> None:
        """
        Charge automatiquement le meilleur modele disponible.

        Priorite : RL (si dispo) > ML
        Le meilleur est determine par Sharpe ratio sur validation 2023.
        """
        with self._lock:
            logger.info("Chargement du meilleur modele...")

            # Pipeline features
            self._pipeline = self._load_feature_pipeline()

            # Determine meilleur modele via metriques
            best_type = self._select_best_model_type()

            if best_type == 'rl':
                try:
                    self._model, self._model_info = self._load_rl_model(version)
                    logger.success(f"Modele actif : RL DQN")
                    return
                except Exception as e:
                    logger.warning(f"RL non dispo ({e}), fallback ML")

            # ML par defaut
            self._model, self._model_info = self._load_ml_model(version)
            logger.success(
                f"Modele actif : ML {self._model_info.name}"
            )

    def _select_best_model_type(self) -> str:
        """
        Compare Sharpe RL vs ML sur validation pour choisir le meilleur.
        """
        sharpe_ml = -999.0
        sharpe_rl = -999.0

        # Sharpe ML (depuis financial_val dans ml_results.json)
        ml_path = Path('evaluation/ml_results.json')
        if ml_path.exists():
            with open(ml_path, 'r') as f:
                data = json.load(f)
            fin_val = data.get('financial_val', {})
            if fin_val:
                sharpes = [
                    v.get('sharpe_ratio', -999)
                    for v in fin_val.values()
                ]
                sharpe_ml = max(sharpes) if sharpes else -999.0

        # Sharpe RL (depuis training log)
        rl_log = Path('evaluation/rl_training_log.json')
        if rl_log.exists():
            with open(rl_log, 'r') as f:
                log = json.load(f)
            if log:
                sharpe_rl = max(e.get('val_sharpe', -999) for e in log)

        logger.info(
            f"Sharpe val -> ML: {sharpe_ml:.4f} | RL: {sharpe_rl:.4f}"
        )

        return 'rl' if sharpe_rl > sharpe_ml else 'ml'

    def get_model(self):
        """Retourne le modele et ses infos (thread-safe)."""
        with self._lock:
            if self._model is None:
                raise RuntimeError(
                    "Aucun modele charge. Appelle load_best_model() d'abord."
                )
            return self._model, self._model_info

    def get_pipeline(self):
        """Retourne le feature pipeline."""
        return self._pipeline

    def list_available_models(self) -> Dict:
        """Liste tous les modeles disponibles dans models/."""
        available: Dict[str, Dict] = {}

        for version_dir in self.models_dir.iterdir():
            if not version_dir.is_dir():
                continue

            version = version_dir.name
            available[version] = {
                'ml': [],
                'rl': [],
            }

            for f in version_dir.glob('ml_*.pkl'):
                available[version]['ml'].append(f.name)

            for f in version_dir.glob('rl_*.pth'):
                available[version]['rl'].append(f.name)

        return available

    @property
    def is_loaded(self) -> bool:
        """Verifie si un modele est charge."""
        return self._model is not None

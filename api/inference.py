"""
Moteur d'inference : normalise les features et predit.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from features.pipeline import FEATURE_COLUMNS

ACTION_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


class InferenceEngine:
    """
    Moteur d'inference unifie pour ML et RL.

    Gere :
    - Validation des features en entree
    - Normalisation via pipeline sauvegarde
    - Inference ML (sklearn/lgbm) ou RL (DQN)
    - Calcul de la confiance

    Usage:
        engine = InferenceEngine(model, model_info, pipeline)
        result = engine.predict(features_dict)
    """

    def __init__(self, model, model_info, pipeline=None) -> None:
        self.model = model
        self.model_info = model_info
        self.pipeline = pipeline
        self.model_type = model_info.type  # 'ml' ou 'rl'

    def validate_features(
        self,
        features: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """
        Valide que toutes les features requises sont presentes.

        Returns:
            (is_valid, missing_features)
        """
        missing = [f for f in FEATURE_COLUMNS if f not in features]
        return len(missing) == 0, missing

    def _prepare_features(
        self,
        features: Dict[str, float],
    ) -> np.ndarray:
        """
        Prepare le vecteur de features pour l'inference.

        - Remplace NaN par 0
        - Applique normalisation si pipeline dispo
        - Retourne vecteur numpy ordonne selon FEATURE_COLUMNS
        """
        # Vecteur ordonne
        feature_vector = np.array(
            [features.get(col, 0.0) for col in FEATURE_COLUMNS],
            dtype=np.float32,
        )

        # Remplace NaN/Inf
        feature_vector = np.nan_to_num(
            feature_vector,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        return feature_vector

    def predict_ml(
        self,
        feature_vector: np.ndarray,
    ) -> Dict[str, Any]:
        """Inference avec modele ML sklearn/lgbm."""
        X = pd.DataFrame(
            [feature_vector],
            columns=FEATURE_COLUMNS,
        )

        proba = self.model.predict_proba(X)[0]

        # Classe 1 = UP -> BUY, Classe 0 = DOWN -> SELL
        if proba[1] >= 0.55:
            action = "BUY"
            confidence = float(proba[1])
        elif proba[1] <= 0.45:
            action = "SELL"
            confidence = float(proba[0])
        else:
            action = "HOLD"
            confidence = float(1 - abs(proba[1] - 0.5) * 2)

        return {
            'action': action,
            'confidence': round(confidence, 4),
            'proba_up': round(float(proba[1]), 4),
            'proba_down': round(float(proba[0]), 4),
        }

    def predict_rl(
        self,
        feature_vector: np.ndarray,
        position: int = 0,
        pnl_unrealized: float = 0.0,
        drawdown: float = 0.0,
    ) -> Dict[str, Any]:
        """Inference avec agent RL DQN."""
        import torch

        # State = features + [position, pnl_unrealized, drawdown]
        state = np.concatenate([
            feature_vector,
            [float(position), float(pnl_unrealized), float(drawdown)],
        ]).astype(np.float32)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(
                self.model.device
            )
            q_values = self.model.policy_net(state_t)[0]
            q_numpy = q_values.cpu().numpy()

        action_idx = int(np.argmax(q_numpy))
        action = ACTION_MAP[action_idx]

        # Confiance = softmax des Q-values
        q_exp = np.exp(q_numpy - q_numpy.max())
        proba = q_exp / q_exp.sum()
        confidence = float(proba[action_idx])

        return {
            'action': action,
            'confidence': round(confidence, 4),
            'q_values': {
                'SELL': round(float(q_numpy[0]), 4),
                'HOLD': round(float(q_numpy[1]), 4),
                'BUY': round(float(q_numpy[2]), 4),
            },
        }

    def predict(
        self,
        features: Dict[str, float],
        position: int = 0,
        pnl_unrealized: float = 0.0,
        drawdown: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Point d'entree principal pour toute inference.

        Args:
            features: Dict feature_name -> valeur
            position: Position actuelle (-1/0/1) pour RL
            pnl_unrealized: PnL non realise pour RL
            drawdown: Drawdown courant pour RL

        Returns:
            Dict avec action, confidence, et metadonnees
        """
        # Validation
        is_valid, missing = self.validate_features(features)
        if not is_valid:
            logger.warning(f"Features manquantes : {missing}")
            # Remplace par 0 les features manquantes
            for f in missing:
                features[f] = 0.0

        # Preparation
        feature_vector = self._prepare_features(features)

        # Inference selon type
        if self.model_type == 'ml':
            result = self.predict_ml(feature_vector)
        else:
            result = self.predict_rl(
                feature_vector, position, pnl_unrealized, drawdown,
            )

        return result

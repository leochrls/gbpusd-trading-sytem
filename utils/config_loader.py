"""Chargement centralisé de la configuration YAML."""

import yaml
from pathlib import Path
from typing import Any


class Config:
    """Charge et expose les paramètres depuis config.yaml.

    Attributes:
        _config: Dictionnaire contenant la configuration complète.
    """

    def __init__(self, config_path: str = None) -> None:
        """Initialise la configuration.

        Args:
            config_path: Chemin vers le fichier config.yaml.
                         Si None, cherche à la racine du projet.
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            self._config: dict = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Récupère une valeur par notation pointée.

        Args:
            key_path: Clé en notation pointée, ex: 'data.raw_dir'.
            default: Valeur par défaut si la clé n'existe pas.

        Returns:
            La valeur correspondante ou default.
        """
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key, default)
            if value is default:
                return default
        return value

    @property
    def raw(self) -> dict:
        """Accès direct au dictionnaire complet."""
        return self._config


# Singleton
config = Config()

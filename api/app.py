"""
API FastAPI pour servir le meilleur modele de trading.

Endpoints :
- POST /predict       : Prediction sur une bougie M15
- GET  /health        : Sante de l'API
- GET  /models/available : Modeles disponibles
- GET  /metrics/latest   : Metriques du modele en prod

REGLE : L'utilisateur NE PEUT PAS relancer l'entrainement.
"""
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from api.inference import InferenceEngine
from api.model_loader import ModelRegistry
from features.pipeline import FEATURE_COLUMNS


# ============================================================
# REGISTRY GLOBAL (singleton)
# ============================================================

registry = ModelRegistry()

# Compteur requetes
request_count = 0
start_time = datetime.now()


# ============================================================
# LIFESPAN (remplace on_event deprecated)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modele au demarrage, cleanup a l'arret."""
    logger.info("=== DEMARRAGE API ===")
    try:
        registry.load_best_model()
        model, info = registry.get_model()
        logger.success(
            f"API prete | Modele : {info.name} ({info.type}) v{info.version}"
        )
    except Exception as e:
        logger.error(f"Erreur chargement modele : {e}")
        raise RuntimeError(f"Impossible de charger le modele : {e}")

    yield

    logger.info("=== ARRET API ===")


# ============================================================
# APP INIT
# ============================================================

app = FastAPI(
    title="GBP/USD Trading System API",
    description=(
        "API de prediction pour le systeme de trading algorithmique GBP/USD.\n\n"
        "- Frequence : M15 (decisions toutes les 15 minutes)\n"
        "- Actions : BUY | SELL | HOLD\n"
        "- Modeles : ML (LightGBM) ou RL (DQN)\n\n"
        "L'entrainement des modeles n'est pas expose via cette API."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SCHEMAS PYDANTIC
# ============================================================

class PredictRequest(BaseModel):
    """Requete de prediction."""

    features: Dict[str, float] = Field(
        ...,
        description="Features de la bougie M15 courante",
        json_schema_extra={
            "example": {
                "return_1": 0.0012,
                "return_4": -0.0005,
                "ema_diff": 0.0003,
                "rsi_14": 52.3,
                "rolling_std_20": 0.0008,
                "range_15m": 0.0025,
                "body_ratio": 0.6,
                "upper_wick_ratio": 0.2,
                "lower_wick_ratio": 0.2,
                "distance_to_ema200": 0.005,
                "slope_ema50": 0.0001,
                "atr_14": 0.0020,
                "rolling_std_100": 0.0007,
                "volatility_ratio": 1.14,
                "adx_14": 25.3,
                "macd": 0.0002,
                "macd_signal": 0.0001,
                "macd_histogram": 0.0001,
            }
        },
    )
    position: int = Field(
        default=0,
        description="Position actuelle : -1=SHORT, 0=FLAT, 1=LONG",
        ge=-1,
        le=1,
    )
    pnl_unrealized: float = Field(
        default=0.0,
        description="PnL non realise en % (pour RL)",
    )
    drawdown: float = Field(
        default=0.0,
        description="Drawdown courant en % (pour RL)",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Version du modele (None = latest)",
    )

    @field_validator('position')
    @classmethod
    def validate_position(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            raise ValueError("position doit etre -1, 0 ou 1")
        return v


class PredictResponse(BaseModel):
    """Reponse de prediction."""

    action: str = Field(..., description="BUY | SELL | HOLD")
    confidence: float = Field(..., description="Confiance [0, 1]")
    model_used: str
    model_type: str
    model_version: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Reponse health check."""

    status: str
    model_loaded: bool
    model_name: str
    model_type: str
    uptime_seconds: float
    total_requests: int
    timestamp: str


# ============================================================
# MIDDLEWARE - LOGGING REQUETES
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count
    request_count += 1

    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} "
        f"({duration * 1000:.1f}ms)"
    )
    return response


# ============================================================
# ENDPOINTS
# ============================================================

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Prediction trading GBP/USD",
    tags=["Prediction"],
)
async def predict(request: PredictRequest):
    """
    Genere une decision de trading pour une bougie M15.

    **Input** : Features techniques de la bougie courante
    **Output** : Action (BUY/SELL/HOLD) + confiance

    Les features doivent etre calculees UNIQUEMENT sur le passe.
    """
    try:
        model, model_info = registry.get_model()
        pipeline = registry.get_pipeline()

        engine = InferenceEngine(model, model_info, pipeline)

        result = engine.predict(
            features=request.features,
            position=request.position,
            pnl_unrealized=request.pnl_unrealized,
            drawdown=request.drawdown,
        )

        return PredictResponse(
            action=result['action'],
            confidence=result['confidence'],
            model_used=model_info.name,
            model_type=model_info.type,
            model_version=model_info.version,
            timestamp=datetime.now().isoformat(),
            details={
                k: v for k, v in result.items()
                if k not in ('action', 'confidence')
            },
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur prediction : {e}")
        raise HTTPException(
            status_code=500, detail=f"Erreur interne : {e}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Sante de l'API",
    tags=["System"],
)
async def health():
    """Verifie que l'API et les modeles sont operationnels."""
    uptime = (datetime.now() - start_time).total_seconds()

    if not registry.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modele non charge",
        )

    _, info = registry.get_model()

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=info.name,
        model_type=info.type,
        uptime_seconds=round(uptime, 2),
        total_requests=request_count,
        timestamp=datetime.now().isoformat(),
    )


@app.get(
    "/models/available",
    summary="Modeles disponibles",
    tags=["Models"],
)
async def list_models():
    """Liste tous les modeles disponibles dans models/."""
    available = registry.list_available_models()
    _, current_info = registry.get_model()

    return {
        "available": available,
        "current": {
            "name": current_info.name,
            "version": current_info.version,
            "type": current_info.type,
            "loaded_at": current_info.loaded_at,
        },
    }


@app.get(
    "/metrics/latest",
    summary="Metriques du modele en production",
    tags=["Models"],
)
async def get_metrics():
    """Retourne les metriques de performance du modele actif."""
    _, info = registry.get_model()

    return {
        "model_name": info.name,
        "model_type": info.type,
        "model_version": info.version,
        "metrics": info.metrics,
        "timestamp": datetime.now().isoformat(),
    }


@app.get(
    "/features/required",
    summary="Liste des features requises",
    tags=["System"],
)
async def get_required_features():
    """Retourne la liste des features attendues par /predict."""
    return {
        "features": FEATURE_COLUMNS,
        "count": len(FEATURE_COLUMNS),
        "description": "Features M15 calculees sur le passe uniquement",
    }


@app.get("/", tags=["System"])
async def root():
    """Point d'entree racine."""
    return {
        "name": "GBP/USD Trading System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }

"""Model catalog: per-provider default models, pricing and context windows.

Bundled defaults live in `backend/model_catalog.yaml`. A provisioned override
file (MEDIASAGE_MODELS_FILE, default `config/models.yaml`) is deep-merged on top,
so deployments can change models or prices without rebuilding the image.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Imported before backend.config loads .env, so load it here too
load_dotenv()

BUNDLED_CATALOG_PATH = Path(__file__).with_name("model_catalog.yaml")
DEFAULT_OVERRIDE_PATH = Path("config/models.yaml")


class ProviderModels(BaseModel):
    """Default models for one provider."""

    analysis: str = ""
    generation: str = ""


class ModelInfo(BaseModel):
    """Pricing and context window for one model."""

    input_cost: float = Field(ge=0)
    output_cost: float = Field(ge=0)
    context_window: int = Field(ge=512)


class ModelCatalog(BaseModel):
    """Full model catalog."""

    providers: dict[str, ProviderModels] = {}
    models: dict[str, ModelInfo] = {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_override_path() -> Path:
    """Path of the provisioned catalog override file."""
    return Path(os.environ.get("MEDIASAGE_MODELS_FILE") or DEFAULT_OVERRIDE_PATH)


def load_model_catalog(override_path: Path | None = None) -> ModelCatalog:
    """Load the bundled catalog merged with the provisioned override file."""
    data = _read_yaml(BUNDLED_CATALOG_PATH)

    path = override_path or get_override_path()
    if path.exists():
        data = _deep_merge(data, _read_yaml(path))
        logger.info("Loaded model catalog overrides from %s", path)
    elif os.environ.get("MEDIASAGE_MODELS_FILE"):
        logger.warning("MEDIASAGE_MODELS_FILE points to missing file %s", path)

    return ModelCatalog.model_validate(data)


CATALOG = load_model_catalog()

# Legacy-shaped lookup tables used across the backend
MODEL_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama": {"analysis": "", "generation": ""},  # Populated from Ollama API
    "custom": {"analysis": "", "generation": ""},  # User-specified
    **{name: p.model_dump() for name, p in CATALOG.providers.items()},
}
MODEL_COSTS: dict[str, dict[str, float]] = {
    name: {"input": m.input_cost, "output": m.output_cost}
    for name, m in CATALOG.models.items()
}
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    name: m.context_window for name, m in CATALOG.models.items()
}

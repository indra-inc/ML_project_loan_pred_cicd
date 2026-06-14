"""
config.py

Responsibility:
- Load environment variables from .env
- Validate required runtime settings
- Provide a single config object for FastAPI app

This file is for serving-time configuration.
Training configs remain inside configs/*.yaml.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# ---------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


@dataclass(frozen=True)
class AppConfig:
    """
    Application runtime configuration.

    frozen=True means values cannot be changed accidentally
    after the config object is created.
    """

    app_env: str
    app_name: str
    app_version: str

    mlflow_tracking_uri: str
    registered_model_name: str
    model_version: Optional[str]
    model_alias: Optional[str]

    target_positive_label: str
    target_negative_label: str


def _get_required_env(key: str) -> str:
    """
    Read a required environment variable.
    Raise error if missing or empty.
    """

    value = os.getenv(key)

    if value is None or value.strip() == "":
        raise ValueError(f"Required environment variable missing: {key}")

    return value.strip()


def _get_optional_env(key: str) -> Optional[str]:
    """
    Read an optional environment variable.
    Return None if missing or empty.
    """

    value = os.getenv(key)

    if value is None or value.strip() == "":
        return None

    return value.strip()


def load_app_config() -> AppConfig:
    """
    Load and validate app config from environment variables.
    """

    try:
        app_config = AppConfig(
            app_env=_get_required_env("APP_ENV"),
            app_name=_get_required_env("APP_NAME"),
            app_version=_get_required_env("APP_VERSION"),

            mlflow_tracking_uri=_get_required_env("MLFLOW_TRACKING_URI"),
            registered_model_name=_get_required_env("REGISTERED_MODEL_NAME_RF"),
            model_version=_get_optional_env("MODEL_VERSION_RF"),
            model_alias=_get_optional_env("MODEL_ALIAS_RF"),

            target_positive_label=_get_required_env("TARGET_POSITIVE_LABEL"),
            target_negative_label=_get_required_env("TARGET_NEGATIVE_LABEL"),
        )

        if app_config.model_version is None and app_config.model_alias is None:
            raise ValueError(
                "Either MODEL_VERSION or MODEL_ALIAS must be provided."
            )

        return app_config

    except Exception as e:
        raise RuntimeError(f"Error while loading app config: {e}")


# Single config object to import elsewhere
app_config = load_app_config()


def main():
    """
    Smoke test:
        python -m app.config
    """

    try:
        print("App config loaded successfully.")
        print(app_config)

    except Exception as e:
        print(f"App config loading failed: {e}")


if __name__ == "__main__":
    main()

# Run: python -m app.config
# Smoke test for app.config
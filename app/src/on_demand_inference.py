from typing import Optional
from dotenv import load_dotenv
from ual.get_config import get_config
from ual.logging import get_logger

from app.src.inference_factory import create_inference_service, validate_model_config

load_dotenv()
logging = get_logger()


def run_on_demand_inference(
    config_path: str, model_name: Optional[str] = None
) -> None:
    """
    Run inference on-demand for a specified time range.

    Supports two modes:
    1. Standalone config: config_path points to a complete config file
    2. Registry-based: config_path points to a registry, model_name selects which model

    Args:
        config_path: Path to YAML config file (standalone or registry)
        model_name: Optional model name to select from registry

    Standalone config should contain:
        - start_time: ISO 8601 UTC timestamp
        - stop_time: ISO 8601 UTC timestamp
        - inputs: List of sensor fields to query
        - targets: List of prediction targets
        - model_name: MLflow model name
        - model_version: MLflow model version
        - sensor_bucket: InfluxDB bucket name
        - sensor_name: Sensor identifier
        - mqtt_topic: MQTT topic for publishing predictions
    """
    logging.info(f"Loading configuration from {config_path}")
    config: dict = get_config(config_path)

    # Check if this is a registry config or standalone config
    if "models" in config:
        # Registry-based mode
        if not model_name:
            logging.error(
                "model_name must be specified when using a models registry config"
            )
            raise ValueError(
                "model_name parameter required when config contains 'models' registry"
            )

        models = config["models"]
        model_config = next(
            (m for m in models if m.get("name") == model_name), None
        )

        if not model_config:
            available_models = [m.get("name") for m in models]
            logging.error(
                f"Model '{model_name}' not found in registry. Available: {available_models}"
            )
            raise ValueError(
                f"Model '{model_name}' not found. Available: {available_models}"
            )

        logging.info(f"Selected model '{model_name}' from registry")
    else:
        # Standalone config mode
        model_config = config

    # Validate time range fields
    if "start_time" not in model_config or "stop_time" not in model_config:
        logging.error(
            "Missing start_time or stop_time in configuration for on-demand inference"
        )
        raise ValueError(
            "On-demand inference requires start_time and stop_time in config"
        )

    # Validate model configuration
    validate_model_config(model_config)

    # Create inference service using factory
    inference_service = create_inference_service(model_config)

    # Run inference for the configured time range
    logging.info(
        f"Starting on-demand inference from {model_config['start_time']} to {model_config['stop_time']}"
    )
    inference_service.initial_inference()
    logging.info("On-demand inference completed successfully")


if __name__ == "__main__":
    # Configure the path to your inference config file here
    # Option 1: Use standalone config file
    CONFIG_PATH: str = "./config/run_config.yaml"
    MODEL_NAME: Optional[str] = None

    # Option 2: Use models registry (uncomment to use)
    # CONFIG_PATH: str = "./models_registry.yaml"
    # MODEL_NAME: Optional[str] = "ual_3_no2"

    run_on_demand_inference(CONFIG_PATH, MODEL_NAME)

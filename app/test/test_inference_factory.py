from ual.get_config import get_config


def test_create_inference_service():
    registry_config: dict = get_config("./config/models_registry.yaml")

    
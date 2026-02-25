from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
from ual.get_config import get_config
from ual.logging import get_logger
from app.src.inference_factory import create_inference_service, validate_model_config
from app.src.time_service import get_next_full_hour

load_dotenv()
logging = get_logger()


if __name__ == "__main__":
    # Load models registry
    registry_config: dict = get_config("./models_registry.yaml")
    models = registry_config.get("models", [])

    if not models:
        logging.error("No models found in models_registry.yaml")
        exit(1)

    logging.info(f"Found {len(models)} model(s) in registry")

    # Validate all model configurations
    for model_config in models:
        validate_model_config(model_config)

    # Create inference services for all models
    services = []
    for model_config in models:
        service = create_inference_service(model_config)
        services.append((model_config["name"], service))

    next_full_hour: datetime = get_next_full_hour()
    logging.info(f"Starting hourly inference schedulers. Next run at: {next_full_hour}")

    # Create schedulers for each service
    schedulers = []
    for i, (model_name, service) in enumerate(services):
        # Use BackgroundScheduler for all but the last service
        if i < len(services) - 1:
            scheduler = BackgroundScheduler()
        else:
            # Use BlockingScheduler for the last service to keep main thread alive
            scheduler = BlockingScheduler()

        scheduler.add_job(
            service.hourly_inference,
            "interval",
            hours=1,
            next_run_time=next_full_hour,
            id=f"hourly_inference_{model_name}",
        )
        schedulers.append((model_name, scheduler))

    # Start all schedulers
    for model_name, scheduler in schedulers:
        logging.info(f"Starting scheduler for model: {model_name}")
        scheduler.start()

    # The last scheduler is blocking, so we'll only reach here on shutdown
    logging.info("All schedulers stopped")

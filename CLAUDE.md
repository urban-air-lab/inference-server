# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style and Formatting

Human Readability is important!
MUST use meaningful, descriptive variable and function names
MUST follow PEP 8 style guidelines
MUST use 4 spaces for indentation (never tabs)
NEVER use emoji, or unicode that emulates emoji (e.g. ✓, ✗). The only exception is when writing tests and testing the impact of multibyte characters.
Use snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
Limit line length to 88 characters (ruff formatter standard)
MUST use type hints for all function signatures (parameters and return values)
NEVER use Any type unless absolutely necessary

## Function Design

MUST keep functions focused on a single responsibility
NEVER use mutable objects (lists, dicts) as default argument values
Limit function parameters to 5 or fewer
Return early to reduce nesting

## Class Design

MUST keep classes focused on a single responsibility
MUST keep __init__ simple; avoid complex logic
Use dataclasses for simple data containers
Prefer composition over inheritance
Avoid creating additional class functions if they are not necessary
Use @property for computed attributes

## Testing

MUST write unit tests for all new functions and classes
MUST mock external dependencies (APIs, databases, file systems)
MUST use pytest as the testing framework
NEVER run tests you generate without first saving them as their own discrete file
NEVER delete files created as a part of testing.
Ensure the folder used for test outputs is present in .gitignore
Follow the Arrange-Act-Assert pattern
Do not commit commented-out tests

## Python Best Practices

NEVER use mutable default arguments
MUST use context managers (with statement) for file/resource management
MUST use is for comparing with None, True, False
MUST use f-strings for string formatting
Use list comprehensions and generator expressions
Use enumerate() instead of manual counter variables

## Security

NEVER store secrets, API keys, or passwords in code. Only store them in .env.
Ensure .env is declared in .gitignore.
NEVER print or log URLs to console if they contain an API key.
MUST use environment variables for sensitive configuration
NEVER log sensitive information (passwords, tokens, PII)




## Project Overview

This is an inference server for Urban Air Lab sensor data that performs machine learning inference on air quality measurements. The system uses a sliding window approach: it loads a trained model from MLflow and runs hourly predictions on sensor data from InfluxDB, publishing results to MQTT.

## Development Commands

### Setup
```bash
# Install dependencies (requires uv package manager)
uv sync --locked

# Create .env file with credentials (get from supervisor)
# Required variables: INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG,
#                    MQTT_SERVER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
#                    MLFLOW_URL, MLFLOW_USERNAME, MLFLOW_PASSWORD
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest app/test/inference_test.py
```

### Running the Application
```bash
# Run locally
uv run app/src/inference.py

# Build Docker image
docker build -t inference-server .

# Run Docker container
docker run --env-file .env inference-server
```

## Architecture

### Core Components

**InferenceService** (`app/src/inference.py`): Main orchestrator that:
- Connects to InfluxDB to query sensor data (via ual.influx.InfluxDBConnector)
- Loads ML models from MLflow (via MLFlowClient)
- Runs initial inference on historical data range specified in run_config.yaml
- Schedules hourly inference jobs using APScheduler (runs at top of each hour + 1 minute buffer)
- Publishes predictions to MQTT broker

**Data Flow**:
1. Query sensor data from InfluxDB (bucket: UAL_MINUTE_CALIBRATION_BUCKET, sensor: UAL-3)
2. Process data using ual.data_processor.DataProcessor:
   - Convert to hourly aggregates
   - Remove NaN values
   - Calculate W-A (working electrode - auxiliary electrode) differences for NO, NO2, O3
   - Align dataframes by timestamp
3. Load scikit-learn model from MLflow
4. Run predictions
5. Publish results to MQTT topic `sensors/ual-hour-inference/ual-3`

**MLFlowClient** (`app/src/mlflow_service.py`): Handles authentication and model loading from MLflow tracking server.

**Time Service** (`app/src/time_service.py`): Utility functions for scheduling:
- `get_last_hour()`: Returns ISO 8601 UTC timestamps for the previous hour window
- `get_next_full_hour()`: Calculates next hour + 1 minute for scheduler start time

### Configuration

**run_config.yaml** (`app/src/run_config.yaml`): Defines inference parameters:
- `start_time`/`stop_time`: Initial inference window
- `inputs`: Sensor fields to query (RAW_ADC values for NO2, NO, O3 electrodes + temperature/humidity)
- `targets`: Prediction targets (currently NO2)
- `model_name`/`model_version`: MLflow model identifier

### Dependencies

- **ual**: Internal Urban Air Lab common library (from github.com/urban-air-lab/common)
  - Provides InfluxDB connectors, MQTT client, data processors, config loaders
- **mlflow**: Model registry and loading
- **apscheduler**: Job scheduling for hourly inference
- **scikit-learn**: ML model support
- **pandas/numpy**: Data manipulation

### Deployment

Docker image built and pushed to GitHub Container Registry (ghcr.io) on main branch commits. CI/CD workflow runs tests before building. Application runs with timezone set to Europe/Berlin.

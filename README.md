# inference-server
Application to run inference on ual sensor data. Currently, (2025-09-01) only a sliding window inference approach is 
implemented. Which runs training on a time windows of data and inference on the next window, just to move forward to the data

## Setup
The projects dependencies are managed with uv. To install and get more information about uv, follow this documentation:
```
https://docs.astral.sh/uv/getting-started/installation/
```

To install the dependencies run

```
uv sync --locked
```

To connect to UrbanAirLabs InfluxDB or Mosquitto (MQTT Broker) the clients need a .env file containing the necessary 
credentials and route information e.g. domain, port, etc. You can get this information from you Supervisor. 

## Run Tests
Tests are base on Pytest - run all tests via command line:

```
uv run pytest
```

## Update UAl Commons
To update changes in UAL commons library use this commands:
```
uv lock --upgrade-package ual
uv sync
```

## Use Ruff for linting
fix linting errors
```
ruff check --fix .
```

format code
```
ruff format . 
```

## Use MyPy for static type checking: 

```
mypy .
```

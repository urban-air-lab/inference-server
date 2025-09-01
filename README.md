# inference-server
Application to run inference on ual sensor data. Currently, (2025-09-01) only a sliding window inference approach is 
implemented. Which runs training on a time windows of data and inference on the next window, just to move forward to the data

## Setup
The projects dependencies are managed with pipenv. To set up the project it´s needed to install pipenv via: 

````pip3 install pipenv````

To install the dependencies run

````pipenv install````

To connect to UrbanAirLabs InfluxDB or Mosquitto (MQTT Broker) the clients need a .env file containing the necessary 
credentials and route information e.g. domain, port, etc. 

### update ual commons 
since pipenv update seems to not work, lib has to be deleted and freshly installed 

````pipenv uninstall git+https://github.com/urban-air-lab/common.git@main#egg=ual````

````pipenv install git+https://github.com/urban-air-lab/common.git@main#egg=ual````


## Run Tests
Unittest are written in pytest. To run all unittest of the project use:

````pipenv run pytest````
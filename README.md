## A Comprehensive Evaluation Framework for Federated Learning

## 1. Install 

#### (0) Environments

1. Operating System : Ubuntu 1804, Mac OS

2. Hardware: Recommend memory 16GB+

#### (1) Build docker image

We first build the docker image that will be used by the server and clients. It may take about 30 minutes, which demends on your network.

```shell script
cd docker
sudo docker build . -t fleval:v1
```

## 2. A QuickStart

#### (1) Generate the data and docker-compose file

```shell script
sudo docker run --rm -v $(pwd):$(pwd) -w $(pwd) fleval:v1 sh -c "python3 4_distribute_data.py && python3 5_generate_docker_compose.py"
```

#### (2) Start the Experiment

```shell script
sudo docker-compose up -d
```

#### (3) View the results

Go to http://127.0.0.1:8200/dashboard to see the results.

Here's an example of the [dashboard](./dashboard.png).

#### (4) Stop the Experiment

Stop and remove the containers

```shell script
sudo docker-compose stop
sudo docker-compose rm
```

Manual stop and remove all the containers

```shell script
sudo docker stop $(sudo docker ps --filter ancestor=fleval:v1 -aq)
sudo docker rm $(sudo docker ps --filter ancestor=fleval:v1 -aq)
```

#### 3. More Experiments

Please take a look at the following three files:

- 1_data_config.yml

- 2_model_config.yml

- 3_runtime_config.yml

All the parameters are in these three files, and you can perform different trials by varying the parameters.
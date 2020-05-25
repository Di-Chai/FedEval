## Install

#### Environments

1. Operating System : Ubuntu 18.04 or Mac OS

2. Hardware: Recommend memory 16GB+

#### Build docker image

We first build the docker image that will be used by the server and clients. It may take about 30 minutes, which demends on your network.

```shell script
cd docker
sudo docker build . -t fleval:v1
```


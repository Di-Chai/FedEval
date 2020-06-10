## Install

#### Environments

1. Operating System : Ubuntu 18.04, Mac OS

2. Hardware: the following table shows the recommended memory size in different dataset and models.

|    Setting    |  Required RAM size   |
| :-----------: | :------------------: |
|   MNIST MLP   | 40 GB / 100 Clients  |
|  MNIST LeNet  | 80 GB / 100 Clients  |
| FEMNIST LeNet | 60 GB / 100 Clients  |
| CelebA LeNet  | 200 GB / 100 Clients |

#### Build docker image

Firstly, please make sure that docker is installed and it can run in root-less mode. More detail can be found here: https://docs.docker.com/engine/security/rootless/

We first build the docker image that will be used by the server and clients. It may take about 30 minutes, which depends on your network.

```shell script
cd docker
docker build . -t fleval:v1
```


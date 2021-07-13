## Install

#### Environments

1. Operating System : Ubuntu 18.04 / 20.04

2. Hardware: the following table shows the recommended memory size in different dataset and models.

|    Setting    |  Required RAM size   |
| :-----------: | :------------------: |
|   MNIST MLP   | 32 GB / 100 Clients  |
|  MNIST LeNet  | 80 GB / 100 Clients  |
| FEMNIST LeNet | 60 GB / 100 Clients  |
| CelebA LeNet  | 200 GB / 100 Clients |

If your compute does not have enough memory, increasing the swap space is a good solution.

#### Build docker image

Firstly, please make sure that docker and docker-compose are properly installed, and the `sudo` commend can run without password.

Then We build the docker image that is used by the server and clients. It may take about 30 minutes, which depends on your network.

```shell script
sudo docker build . -t fedeval:v1
```
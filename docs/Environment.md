## Environment

In this section, we will help you building the environment. 

### Software Dependencies

#### System Requirements

Currently, this system only support **Linux** platforms.

Recommended memory sizes for different dataset and models are shown in the following table. Increasing the swap space is a good solution if your compute does not have enough memory.

| Dataset | ML Model |  Required RAM size   |
| :-----: | :------: | :------------------: |
|  MNIST  |   MLP    | 32 GB / 100 Clients  |
|  MNIST  |  LeNet   | 80 GB / 100 Clients  |
| FEMNIST |  LeNet   | 60 GB / 100 Clients  |
| CelebA  |  LeNet   | 128 GB / 100 Clients |
| Sent140 |  LSTM    | 128 GB / 100 Clients |

#### Docker & Docker Compose (without sudo)

[Docker](https://www.docker.com/) is an indispensable component to virtualize the software and network environment in order to isolate each instance. Follow these [instructions](https://docs.docker.com/engine/install/) to install Docker on your machine. Follow the guideline on [Link](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) to manage docker as non-root user. Docker has been successfully installed if you got a similar output like this:

```shell
$ docker --version
Docker version 20.10.2, build 2291f61
```

[Docker Compose](https://docs.docker.com/compose/) is a tool aimed at easing multi-container operations. Install it under the [guidance](https://docs.docker.com/compose/install/) and check it with:

```shell
$ docker-compose version
docker-compose version 1.27.4, build 40524192
docker-py version: 4.3.1
CPython version: 3.7.4
OpenSSL version: OpenSSL 1.1.1c  28 May 2019
```

#### Python

Though most of the programs can be boosted inside containers, it is still recommended to have a Python interpreter installed on your machine for the convenience and immediacy of type checking, auto complete, and so on during code profiling or some advanced, self-defined operations.

For the sake of compatibility with main components of the system, the version of the interpreter should be in the range of 3.5–3.8 currently. You can find how to install them [here](https://www.python.org/downloads/).

#### Git (Optional)

[Git](https://git-scm.com/) is a distributed version control system. We use it and host this project on [Github](https://github.com/Di-Chai/FedEval). Feel free to skip this because you also can download the project as a zip file from the web page.

Clone the project from Github in this way if have installed git.

```shell
$ git clone https://github.com/Di-Chai/FedEval.git -b dev-0.1
```

### Build the Docker Image

The image we build will be used by each instance (both server and clients).

First, clone the project from GitHub:

```shell
git clone https://github.com/Di-Chai/FedEval.git
```

Enter the project directory and build the image.

```shell
cd docker
sudo docker build . -t fedeval:v1
```

Check image status. This command will list all docker images on your machine.

```shell
sudo docker image ls
```

And you'll got something like these if nothing goes wrong. It's okay if the image ID is different.

```
REPOSITORY     TAG     IMAGE ID       CREATED         SIZE
fedeval        v1      2386be014ab5   3 months ago    4.39GB
```

### Download the dataset

The dataset is available at : [resources](https://www.jianguoyun.com/p/DZ8I2q8QhdfRChiDntUEIAA)

Download the `data.tar.gz`, and put it at the project directory. Using the following commend to unpack the data:

```bash
mv data.tar.gz FedEval
cd FedEval
tar -zxvf data.tar.gz
```

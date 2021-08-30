FROM tensorflow/tensorflow:2.2.3-gpu-py3

# Install python
RUN apt update && apt install -y python3 python3-pip iproute2 net-tools wget vim nload iftop unzip --fix-missing

# build the tf-wrapper and FedEval
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip

# Install python requirements (using douban image)
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt --no-cache-dir

# Place the dataset in the docker image
RUN mkdir -p /root/.keras
RUN mkdir -p /root/.keras/datasets
WORKDIR /root/.keras/datasets
RUN wget "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
RUN wget 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
RUN wget 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
RUN mv cifar-10-python.tar.gz cifar-10-batches-py.tar.gz
RUN tar -zxvf cifar-10-batches-py.tar.gz
RUN tar -zxvf cifar-100-python.tar.gz

# Place the pre-train model in the docker image
RUN mkdir -p /root/.keras/models
#WORKDIR /root/.keras/models
#RUN wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5"
#RUN wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
#RUN wget 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
#RUN wget 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top.h5'

# Install dependency for FetchSGD
WORKDIR /root/
RUN wget https://github.com/nikitaivkin/csh/archive/master.zip
RUN unzip master.zip
WORKDIR /root/csh-master/
RUN pip install -e .

WORKDIR /root/

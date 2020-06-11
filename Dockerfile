FROM ubuntu:18.04

# Install python
RUN apt update && apt install -y python3 python3-pip iproute2 net-tools wget vim

# build the tf-wrapper and FedEval
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Install python requirements
RUN pip install setuptools matplotlib PyYAML pympler \
                Flask-SocketIO Flask socketIO-client opencv-python-headless \
                numpy keras tensorflow==1.14.0 scipy pillow

ADD FedEval /root/FedEval
ADD tf_wrapper /root/tf_wrapper
WORKDIR /root/FedEval
RUN python setup.py install
WORKDIR /root/tf_wrapper
RUN python setup.py install

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
WORKDIR /root/.keras/models
RUN wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5"
RUN wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
RUN wget 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
RUN wget 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top.h5'

WORKDIR /root/
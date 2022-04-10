FROM armswdev/tensorflow-arm-neoverse:r22.03-tf-2.8.0-onednn-acl

# Install python & tools
RUN apt update && \
    apt install -y iproute2 net-tools wget nload iftop unzip pkg-config --fix-missing && \
    apt clean

# Install python requirements (using douban image)
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/

# Place the dataset in the docker image
RUN mkdir -p /root/.keras && \
    mkdir -p /root/.keras/datasets
WORKDIR /root/.keras/datasets
RUN wget "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz" && \
    wget 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' && \
    wget 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz' && \
    mv cifar-10-python.tar.gz cifar-10-batches-py.tar.gz && \
    tar -zxvf cifar-10-batches-py.tar.gz && \
    tar -zxvf cifar-100-python.tar.gz

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

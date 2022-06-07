FROM ubuntu:18.04

# Change the source of apt
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

RUN apt update && apt install -y cmake git build-essential libssl-dev libgmp-dev libboost-all-dev iproute2 net-tools wget nload iftop unzip --fix-missing

WORKDIR /root/
# RUN git clone -bv0.1 https://github.com/emp-toolkit/emp-tool.git
COPY emp-tool /root/emp-tool
RUN cd emp-tool && cmake . && make install

WORKDIR /root/
RUN git clone -bv0.1 https://github.com/emp-toolkit/emp-ot.git
RUN cd emp-ot && cmake . && make install

WORKDIR /root/
RUN apt install -y wget
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
RUN tar -zxvf eigen-3.3.7.tar.gz
RUN cd eigen-3.3.7 && mkdir build && cd build && cmake .. && make install

WORKDIR /root/
# RUN git clone https://github.com/shreya-28/Secure-ML.git
# RUN cd Secure-ML && mkdir build && cd build && cmake .. && make

COPY Secure-ML /root/Secure-ML
WORKDIR /root/Secure-ML/build
RUN cmake .. && make

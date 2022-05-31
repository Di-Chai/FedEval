FROM tensorflow/tensorflow:latest-gpu

# Install python & tools
RUN apt update && \
    apt install -y iproute2 net-tools wget nload iftop unzip --fix-missing && \
    apt clean

# Install python requirements (using douban image)
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple/

WORKDIR /root/

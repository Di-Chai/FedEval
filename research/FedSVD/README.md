## FedSVD

This is the open-sourced code for paper "Practical Loss Federated Singular Vector Decomposition over Billion-Scale Data".

The source code for implementing FedSVD is located at [FedSVD.py](../../FedEval/strategy/FedSVD.py).

To reproduce the results in the paper, please follow the following steps.

### 1) Prepare the environment

Since we implement and benchmark the FedSVD in [FedEval](https://github.com/Di-Chai/FedEval), please refer to [Environment](https://di-chai.github.io/FedEval/Environment.html) to get the environment ready.

### 2) Reproduce the results

We have provided a script for reproducing all the efficiency and lossless results for SVD and its applications.

The detailed commends are listed bellow.

```bash
cd research/FedSVD

# SVD Precision (Lossless)
python trial.py --mode svd --task precision

# SVD Efficiency
python trial.py --model svd --task latency
python trial.py --mode svd --task bandwidth
python trial.py --mode svd --task very_clients
python trial.py --mode svd --task small_scale
python trial.py --mode svd --task large_scale
python trial.py --mode svd --task block_size

# SVD Application Precision (Lossless)
python trial.py --mode lr --task precision
python trial.py --mode pca --task precision

# SVD Application Efficiency
python trial.py --model lr --task latency
python trial.py --mode lr --task bandwidth
python trial.py --mode lr --task large_scale
python trial.py --mode svd --task large_scale_recsys

# Evaluate the Proposed Optimizations
python trial.py --model svd --task block_size
```

### 3) Attack experiments

We perform the attack experiments using Matlab, and the code could be found here: [Attack (Matlab Code)](ica_attack).

To reproduce the results, please execute the `main.m` file. 

### 4) Baseline methods (SecureML)

The code for reproducing the SecureML baseline results is presented here: [SecureML](baseline_secureml)

Two steps are required to reproduce the results:

```bash
# Step 1, build the docker image
cd research/FedSVD/baseline_secureml
docker build . -t mpc:v1

# Step 2, run the experiments
python trial.py
```

### 5) Baseline methods (FATE)

Preparing the environment for FATE is more complex than SecureML. 
To simplify the reproduction process, we have uploaded the open-to-use environment to DockerHub. Specifically, the environment is prepared using this tutorial: [FATE official guideline for preparing the env](https://github.com/FederatedAI/FATE/blob/master/deploy/cluster-deploy/doc/fate_on_eggroll/fate-allinone_deployment_guide.md).

To reproduce the FATE experiment results, please follow the following guideline:

```bash
# Step 1, pull the docker image
docker pull dichai/fate:host
docker pull dichai/fate:guest

# Step 2, Create the docker network
# Note: Please do not change the subnet since this subnet (192.168.0.0/16) is fixed in the image.
docker network create --subnet=192.168.0.0/16 fate_network

# Step 3, Download the Host/Guest data

# Step 4, Start the Host/Guest Containers
# Note: 1) please change the FATE_HOST_PATH and FATE_GUEST_PATH value accordingly, 
#       2) do not change the fixed ip address of the containers,
#       3) You may need to open two terminals to start the Host and Guest Containers separately.
# In terminal 1:
export FATE_HOST_PATH = ~/fate_data/host
docker run -it -v $FATE_HOST_PATH:/data -w /data --name fate_host --net fate_network --ip 192.168.0.3 --cap-add NET_ADMIN -p 8100:8080 dichai/fate:host bash
# In terminal 2:
export FATE_GUEST_PATH = ~/fate_data/guest
docker run -it -v $FATE_GUEST_PATH:/data -w /data --name fate_guest --net fate_network --ip 192.168.0.4 --cap-add NET_ADMIN -p 8200:8080 dichai/fate:guest bash

# Step 5, Start the FATE services
# In both guest and host containers:
bash start_service.sh  # or restart_service.sh if the containers are stopped and restarted

# Step 6, Generate the synthetic data
# In guest container:
cd /data/synthetic_data
# You may change the value of sample_size & feature_size, 
#   but we recommend that you change it later after the experiments success with the value we provided.
python generate_data.py --sample_size 10000 --feature_size 1000  
# Transfer the host-data to host
scp synthetic_10000_1000_host.csv root@fate_host:/data/synthetic_data
# (Optional) You may delete the host-data at guest by: 
sudo rm *host.csv

# Step 7, Upload the csv data to database
# In both guest and host containers:
cd /data/experiments
flow data upload -c upload_data_10000.json  # If you didn't change the sample_size & feature_size in Step 6, this json-config should work, otherwise you need to modify it accordingly.

# Step 8, Start the experiment using "flow"
# In guest container:
cd /data/experiments
flow job submit -c simplified_conf.json -d simplified_dsl.json  # Again, if you didn't change the sample_size & feature_size in Step 6, this config & dsl should work, otherwise you need to modify it accordingly.

# Step 9, Reproduce the results in the paper
# In guest container:
cd /data/experiments
python trial.py  # This will automatically run all the benchmark experiments which takes a long time, you may modify the scripts to run only part of them.

# Step 10, Check the results
# You may view the experiment status at: ip_of_your_machine:8200 (i.e., the Guest FATEBoard address),
#   The login user and passwd are all "admin"
# If you are running the trial.py, the time consumption of each job will be logged into trial.log
```

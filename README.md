## Benchmark Results Released!

Link to benchmark results: https://di-chai.github.io/FedEval/BenchmarkResults.html

Link to reproduction guidelines: https://di-chai.github.io/FedEval/Reproduce.html

## FedEval (iSpree Model)

----

[![Documentation Status](https://readthedocs.org/projects/fedeval/badge/?version=latest)](https://fedeval.readthedocs.io/en/latest/?badge=latest)

FedEval is a benchmarking platform with comprehensive evaluation model (i.e., iSpree model) for federated learning. The arxiv paper is available at: https://arxiv.org/abs/2011.09655

#### The iSpree Evaluation Model

The iSpree evaluation model is the core of our benchmarking system. It defines six evaluation metrics that cannot be excluded in the system: Incentive, Security, Privacy, Robustness, Efficacy, and Efficiency.

#### Docs

- [User guide and Documentation](https://di-chai.github.io/FedEval/)

#### Introduction to FedEval

![The framework of FedEval benchmarking system](https://di-chai.github.io/FedEval/_images/bm_system.png)

We propose a federated benchmarking system called FedEval shown in the above figure, which demonstrates the inputs, inner architecture, and outputs of the system. Our iSpree evaluation model is built inside the benchmarking system. Three key modules are designed in the system:

-  **Data config and the *data_loader* module**: Our benchmarking system provides four standard federated learning datasets, and different data settings (e.g., non-IID data) can be implemented by changing the data configs. Self-defined data is also supported. We only need to define the *load\_data* function in *data\_loader* module to add a new dataset, which will share the same set of processing functions with the built-in datasets.
-  **Model config and the Keras Model module**: Currently three machine learning models are built inside our system, including MLP, LeNet, and StackedLSTM. We use TensorFlow as the backend, and all the models are made via subclassing the Keras model. Thus adding new machine learning models are very simple in our framework.
-  **Runtime config and the *strategy* module**: One of the essential components in our benchmarking system is the *strategy* module, which defines the protocol of the federated training. Briefly, the FL *strategy* module supports the following customization:
   -  Customized uploading message, i.e., which parameters are uploaded to the server from the clients.
   -  Customized server aggregation method, e.g., weighted average.
   -  Customized training method for clients, e.g., the clients' model can be trained using regular gradient descent method or the other solutions like meta-learning methods.
   -  Customized method for incorporating the global and local model, e.g., one popularly used method is replacing the local model with the global one before training.

We use the docker container technology to simulate the server and clients (i.e., each participant is a container), and use socket IO in the communication. The isolation between different containers guarantees that our simulation can reflect the real-world application. The entire system is open-sourced, seven benchmark FL datasets, including MNIST, CIFAR10, CIFAR100, FEMNIST, CelebA, Semantic140, and Shakespeare. The essential components (i.e., dataset, ML models, and FL strategy) can be easily used or self-defined. Thus researches can implement their new idea and evaluate with iSpree model very quickly.

Briefly, three steps are needed to start an experiment in our benchmarking system:

-  **Step 1**: Determine the benchmark dataset, ML model, and FL strategy, then modify the data, model, and runtime configs based on the templates.

-  **Step 2**: Use the built-in tool to generate data for clients and create the docker-compose files.

-  **Step 3**: Start the experiments using docker-compose, and monitor the dashboard for the evaluation status and results.

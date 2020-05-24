## Dataset

Three datasets are used in the experiments: [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [FEMNIST](https://github.com/TalwalkarLab/leaf#datasets), [CelebA](https://github.com/TalwalkarLab/leaf#datasets). We perform the classification task on all datasets. FEMNIST is an extended MNIST dataset based on handwritten digits and characters. CelebA builds on the Large-scale CelebFaces Attributes Dataset, and we use the *smiling* label as the classification target. For the non-IID evaluation, we use different strategies. In MNIST and CIFAR-10, we simulate non-IID by restricting the number of clients' local image classes. Experiments of each client has 1,2​ and 3 classes are reported. In FEMNIST and CelebA, we have the identity of who generates the data. Thus we partition the data naturally based on the identity, and perform a random shuffle for IID setting.

On average, each client holds 300 images on MNIST and CIFAR-10 datasets, 137 images on the FEMNIST dataset, and 24 images on the CelebA dataset. At each client, we randomly select 80\%, 10\%, and 10\% for training, validation, and testing. 

## Hardware

All the experiments run on a cluster of three machines. One machine with Intel(R) Xeon(R) E5-2620 32-core 2.10GHz CPU, 378GB RAM, and two machines with Intel(R) Xeon(R) E5-2630 24-core 2.6GHz CPU, 63GB RAM. We put the server on the first machine and 40, 30, 30 clients on three machines, respectively. 
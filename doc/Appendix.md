# Appendix

## Appendix 1

#### Datasets and Hardware

Three datasets are used in the experiments: [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [FEMNIST](https://github.com/TalwalkarLab/leaf#datasets), [CelebA](https://github.com/TalwalkarLab/leaf#datasets). We perform the classification task on all datasets. FEMNIST is an extended MNIST dataset based on handwritten digits and characters. CelebA builds on the Large-scale CelebFaces Attributes Dataset, and we use the *smiling* label as the classification target. For the non-IID evaluation, we use different strategies. In MNIST and CIFAR-10, we simulate non-IID by restricting the number of clients' local image classes. Experiments of each client has 1,2​ and 3 classes are reported. In FEMNIST and CelebA, we have the identity of who generates the data. Thus we partition the data naturally based on the identity, and perform a random shuffle for IID setting.

On average, each client holds 300 images on MNIST and CIFAR-10 datasets, 137 images on the FEMNIST dataset, and 24 images on the CelebA dataset. At each client, we randomly select 80\%, 10\%, and 10\% for training, validation, and testing. 

All the experiments run on a cluster of three machines. One machine with Intel(R) Xeon(R) E5-2620 32-core 2.10GHz CPU, 378GB RAM, and two machines with Intel(R) Xeon(R) E5-2630 24-core 2.6GHz CPU, 63GB RAM. We put the server on the first machine and 40, 30, 30 clients on three machines, respectively. 

## Appendix 2

[Grid search result for FedAvg.](https://anonymous.4open.science/repository/19c83e2b-0176-4ae2-b231-b260a74794e3/doc/fedavg_grid_search.md)

## Appendix 3

<table style="text-align:center">
    <tr>
        <td></td>
        <td colspan="3">MNIST LeNet FedAvg</td>
        <td colspan="3">MNIST LeNet FedSGD</td>
        <td colspan="3">MNIST MLP FedAvg</td>
        <td colspan="3">MNIST MLP FedSGD</td>
    </tr>
    <tr>
        <td>LR</td>
        <td>GD</td>
        <td>Momuntum</td>
        <td>Adam</td>
        <td>GD</td>
        <td>Momuntum</td>
        <td>Adam</td>
        <td>GD</td>
        <td>Momuntum</td>
        <td>Adam</td>
        <td>GD</td>
        <td>Momuntum</td>
        <td>Adam</td>
    </tr>
    <tr>
        <td>0.0001</td>
        <td>0.985</td>
        <td>0.991</td>
        <td>0.99</td>
        <td>0.968</td>
        <td>0.798</td>
        <td>0.984</td>
        <td>0.98</td>
        <td>0.981</td>
        <td>0.983</td>
        <td>0.963</td>
        <td>0.986</td>
        <td>0.981</td>
    </tr>
    <tr>
        <td>0.0005</td>
        <td>0.99</td>
        <td>0.995</td>
        <td>0.992</td>
        <td>0.972</td>
        <td>0.612</td>
        <td>0.995</td>
        <td>0.979</td>
        <td>0.982</td>
        <td>0.986</td>
        <td>0.982</td>
        <td>0.983</td>
        <td>0.984</td>
    </tr>
    <tr>
        <td>0.001</td>
        <td>0.988</td>
        <td>0.993</td>
        <td>0.995</td>
        <td>0.979</td>
        <td>0.623</td>
        <td>0.992</td>
        <td>0.982</td>
        <td>0.986</td>
        <td>0.982</td>
        <td>0.984</td>
        <td>0.981</td>
        <td>0.983</td>
    </tr>
    <tr>
        <td>0.005</td>
        <td>0.993</td>
        <td>0.097</td>
        <td>0.99</td>
        <td>0.189</td>
        <td>0.26</td>
        <td>0.993</td>
        <td>0.983</td>
        <td>0.983</td>
        <td>0.974</td>
        <td>0.658</td>
        <td>0.438</td>
        <td>0.982</td>
    </tr>
    <tr>
        <td>0.01</td>
        <td>0.99</td>
        <td>0.113</td>
        <td>0.983</td>
        <td>0.281</td>
        <td>0.373</td>
        <td>0.994</td>
        <td>0.983</td>
        <td>0.113</td>
        <td>0.957</td>
        <td>0.17</td>
        <td>0.184</td>
        <td>0.981</td>
    </tr>
    <tr>
        <td>Central</td>
        <td colspan="6">0.995</td>
        <td colspan="6">0.988</td>
    </tr>
    <tr>
        <td></td>
        <td colspan="3">FEMNIST LeNet FedAvg</td>
        <td colspan="3">FEMNIST LeNet FedSGD</td>
        <td colspan="3">FEMNIST MLP FedAvg</td>
        <td colspan="3">FEMNIST MLP FedSGD</td>
    </tr>
    <tr>
        <td>LR</td>
        <td>GD</td>
        <td>Momuntum</td>
        <td>Adam</td>
        <td>GD</td>
        <td>Momuntum</td>
        <td>Adam</td>
        <td>GD</td>
        <td>Momuntum</td>
        <td>Adam</td>
        <td>GD</td>
        <td>Momuntum</td>
        <td>Adam</td>
    </tr>
    <tr>
        <td>0.0001</td>
        <td>0.74</td>
        <td>0.815</td>
        <td>0.803</td>
        <td>0.332</td>
        <td>0.826</td>
        <td>0.651</td>
        <td>0.719</td>
        <td>0.762</td>
        <td>0.798</td>
        <td>0.51</td>
        <td>0.574</td>
        <td>0.682</td>
    </tr>
    <tr>
        <td>0.0005</td>
        <td>0.792</td>
        <td>0.827</td>
        <td>0.846</td>
        <td>0.743</td>
        <td>0.702</td>
        <td>0.81</td>
        <td>0.779</td>
        <td>0.76</td>
        <td>0.754</td>
        <td>0.299</td>
        <td>0.256</td>
        <td>0.779</td>
    </tr>
    <tr>
        <td>0.001</td>
        <td>0.804</td>
        <td>0.826</td>
        <td>0.843</td>
        <td>0.72</td>
        <td>0.515</td>
        <td>0.846</td>
        <td>0.784</td>
        <td>0.072</td>
        <td>0.541</td>
        <td>0.218</td>
        <td>0.12</td>
        <td>0.814</td>
    </tr>
    <tr>
        <td>0.005</td>
        <td>0.833</td>
        <td>0.262</td>
        <td>0.763</td>
        <td>0.125</td>
        <td>0.067</td>
        <td>0.848</td>
        <td>0.767</td>
        <td>0.067</td>
        <td>0.089</td>
        <td>0.079</td>
        <td>0.079</td>
        <td>0.703</td>
    </tr>
    <tr>
        <td>0.01</td>
        <td>0.84</td>
        <td>0.063</td>
        <td>0.618</td>
        <td>0.094</td>
        <td>0.079</td>
        <td>0.848</td>
        <td>0.558</td>
        <td>0.067</td>
        <td>0.067</td>
        <td>0.079</td>
        <td>0.079</td>
        <td>0.439</td>
    </tr>
    <tr>
        <td>Central</td>
        <td colspan="6">0.85</td>
        <td colspan="6">0.829</td>
    </tr>
</table>


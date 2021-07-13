.. _header-n0:

Appendix 1 The Benchmarking System
==================================

.. raw:: html

   <html xmlns="http://www.w3.org/1999/xhtml">
   <head></head>
   <body><center><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAiIAAAI4CAYAAABX+YLXAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAF9oSURBVHhe7Z09qy3XdqZ9je3Epi0FFzu4wQElNm2wuAhsaAfiJnKoTOnBQcNxJNOBQZEaR0osh0oaJQYlDSexwNCBDPoB0h8Qx3Qs0E/Y3c/WHucOTY36WqvWqllVzwMve++qWbNGzVVV411zzqr9Ow8iIiIiG6ERERERkc3QiIiIiMhmaERERERkMzQiIiIishkaEREREdkMjYiIiIhshkZERERENkMjIiIiIpuhEREREZHN0IiIiIjIZmhEREREZDM0IiIiIrIZGhERERHZDI2IiIiIbIZGRERERDZDIyIiIiKboRERERGRzdCIiIiIyGZoRERERGQzNCIiIiKyGRoRERER2QyNiIiIiGyGRkREREQ2QyMiIiIim6ERERERkc3QiIiIiMhmaERERERkMzQiIiIishkaEREREdkMjYiIiIhshkZERERENkMjIiIiIpuhEREREZHN0IiIiIjIZmhEREREZDM0IiIiIrIZGhERERHZDI2IiIiIbIZGRERERDZDIyIiIiKboRERERGRzdCIiIiIyGZoRERERGQzNCIiIiKyGRoRERER2QyNiIiIiGyGRkREREQ2QyMiIiIim6ERERERkc3QiIiIiMhmaERERERkMzQiIiIishkaEREREdkMjYiIiIhshkZERERENkMjIiIio3zyyScP77zzzsPv/M7vPIrfP/vss4fvv//+cd3XX3/9VHI/ROwffPDB0xLZCo2IiIiUkKzDgITxAIzHe++999qYDBmRL7/88jHZ9wbH8uabbz7GznHItmhERESkBBNBsn7x4sXTkp/C8iEj8t133z0m+56MyBdffPHw1ltvvTZQGpE+0IiIiEhJ9BoMGREgsbdGJPek9GJEIqbck6MR6QONiIjcna+++mo3OjM5YTOcUcHybES+/fbbn8wn6XFo5qOPPnodn0ZkezQiInJ33njjjdeJoGcR55mphjGG5oMAc0KiF6VVm/Apmw0LvS4M5wQxLJSF6cnbMNE05q0sIdetEdkejYiI3B2NyD7ALFTtMmVIctmqR4S5GqzD6GAk6EXhb0xMNhZRLoQJwaxQPgxP1LEEjUhf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" width="50%" style="margin: 0 auto;"></center></body>
   </html>

Having motivated the need for an easy-to-use and lightweight
benchmarking system, we propose a federated benchmarking system shown in
this figure. The system contains the following three steps:

-  **Step 1:** Distribute the data to all clients, where different
   strategies can be applied to generate IID or non-IID data.

-  **Step 2:** Provide the training scripts and distribute the codes to
   all the clients.

-  **Step 3:** Training the model. The ACTPR evaluation model will run
   automatically in this step.

We used the docker container technology to simulate the server and
clients. The isolation between different containers guarantees that our
simulation can reflect the real-world application. All the benchmark
system is open-sourced, and four benchmark FL datasets are included in
the system. Thus researches can implement their new idea and evaluate
with ACTPR model very quickly.

.. _header-n12:

Appendix 2 Dataset, Model settings and Hardware
================================

Three datasets are used in the experiments:
`MNIST <http://yann.lecun.com/exdb/mnist/>`__,
`FEMNIST <https://github.com/TalwalkarLab/leaf#datasets>`__,
`CelebA <https://github.com/TalwalkarLab/leaf#datasets>`__. We perform
the classification task on all datasets. FEMNIST is an extended MNIST
dataset based on handwritten digits and characters. CelebA builds on the
Large-scale CelebFaces Attributes Dataset, and we use the *smiling*
label as the classification target. For the non-IID evaluation, we use
different strategies. In MNIST, we simulate non-IID by
restricting the number of clients' local image classes. Experiments of
each client has 1,2 and 3 classes are reported. In FEMNIST and CelebA,
we have the identity of who generates the data. Thus we partition the
data naturally based on the identity, and perform a random shuffle for
IID setting.

On average, each client holds 300 images on MNIST,
137 images on the FEMNIST dataset, and 24 images on the CelebA dataset.
At each client, we randomly select 80\%, 10\%, and 10\% for training,
validation, and testing.

We use MLP and LeNet models in the experiments. The MLP model is implemented using 2 hidden layers with 512 units, ReLU activation, and dropout probability 0.2. The LeNet model is implemented using the same parameter with the work of LeCun et al. [1].

All the experiments run on a cluster of three machines. One machine with
Intel(R) Xeon(R) E5-2620 32-core 2.10GHz CPU, 378GB RAM, and two
machines with Intel(R) Xeon(R) E5-2630 24-core 2.6GHz CPU, 63GB RAM. We
put the server on the first machine and 40, 30, 30 clients on three
machines, respectively.

[1] LeCun Y, Bottou L, Bengio Y, et al. `Gradient-based learning applied to document recognition[J]. <http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>`_ Proceedings of the IEEE, 1998, 86(11): 2278-2324.  

Appendix 3 Tuning the Optimizer and Learning Rate
=================================================

3.1 Results of tuning the optimizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following table show the results of FL accuracy when we vary the
learning rate (lr) and optimizers. The experiments are performed on
MNIST and FEMNIST datasets, using MLP and LeNet model, FedSGD and FedAvg
FL schemas. Among these three optimizers, **Adam can achieve the best
accuracy more frequently, and Adam is more robust given different lr**.

.. raw:: html

   <html xmlns="http://www.w3.org/1999/xhtml">
   <head>
   <style>
        table{
            box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
        }
        table,th,tr,td{ 
            padding: 7px 7px 7px 7px;
            border-bottom:1px solid #dedede;
            border-left:1px solid #dedede;
            border-top:1px solid #dedede;
            border-right:1px solid #dedede;
            text-align: center;
        }
    </style>
   </head>
   <body>
   <div style="width:100%;display:block;overflow-y:auto;">
   <table style="text-align:center; border-bottom:1px solid #dedede;">
       <tbody><tr>
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
           <td>CentralAcc</td>
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
           <td>CentralAcc</td>
           <td colspan="6">0.85</td>
           <td colspan="6">0.829</td>
       </tr>
   </tbody></table>
   </div>
   </body></html>

--------------

3.2 Results of tuning the learning rate (LR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <html xmlns="http://www.w3.org/1999/xhtml"><head></head><body>
   <div style="width:100%;display:block;overflow-y:auto;">
   <table style="text-align:center">
       <tbody><tr>
           <td></td>
           <td colspan="3">MNIST LeNet FedAvg</td>
           <td colspan="3">MNIST LeNet FedSGD</td>
       </tr>
       <tr>
           <td>LR</td>
           <td>Acc</td>
           <td>CommRound</td>
           <td>TimeAll</td>
           <td>Acc</td>
           <td>CommRound</td>
           <td>TimeAll</td>
       </tr>
       <tr>
           <td>0.0001</td>
           <td>0.99</td>
           <td>174</td>
           <td>13.9</td>
           <td>0.984</td>
           <td>2000</td>
           <td>160.4</td>
       </tr>
       <tr>
           <td>0.0005</td>
           <td>0.992</td>
           <td>113</td>
           <td>8.8</td>
           <td>0.995</td>
           <td>1609</td>
           <td>144</td>
       </tr>
       <tr>
           <td>0.001</td>
           <td>0.995</td>
           <td>104</td>
           <td>8.4</td>
           <td>0.992</td>
           <td>1128</td>
           <td>100.7</td>
       </tr>
       <tr>
           <td>0.005</td>
           <td>0.99</td>
           <td>90</td>
           <td>7.3</td>
           <td>0.993</td>
           <td>286</td>
           <td>26.1</td>
       </tr>
       <tr>
           <td>0.01</td>
           <td>0.983</td>
           <td>53</td>
           <td>4.4</td>
           <td>0.994</td>
           <td>157</td>
           <td>14</td>
       </tr>
       <tr>
           <td>CentralAcc</td>
           <td colspan="6">0.995</td>
       </tr>
       <tr>
           <td></td>
           <td colspan="3">MNIST MLP FedAvg</td>
           <td colspan="3">MNIST MLP FedSGD</td>
       </tr>
       <tr>
           <td>LR</td>
           <td>Acc</td>
           <td>CommRound</td>
           <td>TimeAll</td>
           <td>Acc</td>
           <td>CommRound</td>
           <td>TimeAll</td>
       </tr>
       <tr>
           <td>0.0001</td>
           <td>0.983</td>
           <td>132</td>
           <td>22.3</td>
           <td>0.981</td>
           <td>2000</td>
           <td>855.5</td>
       </tr>
       <tr>
           <td>0.0005</td>
           <td>0.986</td>
           <td>81</td>
           <td>13.7</td>
           <td>0.984</td>
           <td>775</td>
           <td>322.4</td>
       </tr>
       <tr>
           <td>0.001</td>
           <td>0.982</td>
           <td>92</td>
           <td>15.8</td>
           <td>0.983</td>
           <td>427</td>
           <td>177.3</td>
       </tr>
       <tr>
           <td>0.005</td>
           <td>0.974</td>
           <td>60</td>
           <td>10.4</td>
           <td>0.982</td>
           <td>141</td>
           <td>59.1</td>
       </tr>
       <tr>
           <td>0.01</td>
           <td>0.957</td>
           <td>56</td>
           <td>9.7</td>
           <td>0.981</td>
           <td>117</td>
           <td>48.9</td>
       </tr>
       <tr>
           <td>CentralAcc</td>
           <td colspan="6">0.988</td>
       </tr>
       <tr>
           <td></td>
           <td colspan="3">FEMNIST LeNet FedAvg</td>
           <td colspan="3">FEMNIST LeNet FedSGD</td>
       </tr>
       <tr>
           <td>LR</td>
           <td>Acc</td>
           <td>CommRound</td>
           <td>TimeAll</td>
           <td>Acc</td>
           <td>CommRound</td>
           <td>TimeAll</td>
       </tr>
       <tr>
           <td>0.0001</td>
           <td>0.803</td>
           <td>120</td>
           <td>10.9</td>
           <td>0.651</td>
           <td>2000</td>
           <td>154</td>
       </tr>
       <tr>
           <td>0.0005</td>
           <td>0.846</td>
           <td>121</td>
           <td>9.4</td>
           <td>0.81</td>
           <td>2000</td>
           <td>153.6</td>
       </tr>
       <tr>
           <td>0.001</td>
           <td>0.843</td>
           <td>115</td>
           <td>8.5</td>
           <td>0.846</td>
           <td>2000</td>
           <td>151.9</td>
       </tr>
       <tr>
           <td>0.005</td>
           <td>0.763</td>
           <td>60</td>
           <td>4.8</td>
           <td>0.848</td>
           <td>618</td>
           <td>46.9</td>
       </tr>
       <tr>
           <td>0.01</td>
           <td>0.618</td>
           <td>72</td>
           <td>6.2</td>
           <td>0.848</td>
           <td>265</td>
           <td>20.6</td>
       </tr>
       <tr>
           <td>CentralAcc</td>
           <td colspan="6">0.85</td>
       </tr>
       <tr>
           <td></td>
           <td colspan="3">FEMNIST MLP FedAvg</td>
           <td colspan="3">FEMNIST MLP FedSGD</td>
       </tr>
       <tr>
           <td>LR</td>
           <td>Acc</td>
           <td>CommRound</td>
           <td>TimeAll</td>
           <td>Acc</td>
           <td>CommRound</td>
           <td>TimeAll</td>
       </tr>
       <tr>
           <td>0.0001</td>
           <td>0.798</td>
           <td>128</td>
           <td>36.8</td>
           <td>0.682</td>
           <td>2000</td>
           <td>441.6</td>
       </tr>
       <tr>
           <td>0.0005</td>
           <td>0.754</td>
           <td>81</td>
           <td>22.5</td>
           <td>0.779</td>
           <td>2000</td>
           <td>447.1</td>
       </tr>
       <tr>
           <td>0.001</td>
           <td>0.541</td>
           <td>74</td>
           <td>18.3</td>
           <td>0.814</td>
           <td>2000</td>
           <td>446.7</td>
       </tr>
       <tr>
           <td>0.005</td>
           <td>0.089</td>
           <td>29</td>
           <td>8.9</td>
           <td>0.703</td>
           <td>571</td>
           <td>126.2</td>
       </tr>
       <tr>
           <td>0.01</td>
           <td>0.067</td>
           <td>25</td>
           <td>7.7</td>
           <td>0.439</td>
           <td>344</td>
           <td>76.1</td>
       </tr>
       <tr>
           <td>CentralAcc</td>
           <td colspan="6">0.829</td>
       </tr>
   </tbody></table>
   </div>
   </body></html>

--------------


Appendix 4 Grid Search on FedAvg
=================================================

.. toctree::
   :maxdepth: 1

   fedavg_grid_search.md


Appendix 5 FC Attack Results
==============================

.. raw:: html


# Secure Machine Learning

Secure Linear Regression in the Semi-Honest Two-Party Setting. More details on the protocol can be found in the [SecureML paper](https://eprint.iacr.org/2017/396.pdf).

### Prerequisites
1. [emp-ot](https://github.com/emp-toolkit/emp-ot/tree/15fb731e528974bcfe5aa09c18bb16376e949283).
2. [Eigen 3.3.7](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download).

### Building Secure-ML
```
git clone https://github.com/shreya-28/Secure-ML.git
cd Secure-ML
mkdir build
cd build
cmake ..
make
```

### Running Secure-ML
The build system creates two binaries, namely, `ideal_functionality` and `secure_ML`. The former represents the functionality that the latter implements securely.  
The binaries can be executed as follows:
* `ideal_functionality`
  - `./build/bin/ideal_functionality [num_iter]`
- `secure_ML`
  - On local machine
    - `./build/bin/secure_ML 1 8000 [num_iter] & ./build/bin/secure_ML 2 8000 [num_iter]`
  - On two different machines
    - `./build/bin/secure_ML 1 8000 [num_iter]` on Machine 1
    - `./build/bin/secure_ML 2 8000 [num_iter] [addr_of_machine_1]` on Machine 2

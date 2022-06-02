#include "read_MNIST.hpp"
#include "linear_regression.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", " << ", ";");

int NUM_IMAGES = BATCH_SIZE;
int PARTY;

int main(int argc, char** argv){
    int port, num_iters;
    string address;

    PARTY = atoi(argv[1]);
    port = atoi(argv[2]);
    num_iters = atoi(argv[3]);

    try{
        int x = -1;
        if(argc <= 4)
            throw x;
        address = argv[4];
    } catch(int x) {
        address = "127.0.0.1";
    }

    NUM_IMAGES *= num_iters;

    NetIO* io = new NetIO(PARTY == ALICE ? nullptr : address.c_str(), port);

    TrainingParams params;

    cout << "========" << endl;
    cout << "Training" << endl;
    cout << "========" << endl;

    vector<vector<uint64_t> > training_data;
    vector<uint64_t> training_labels;

    read_MNIST_data<uint64_t>(true, training_data, params.n, params.d);
    RowMatrixXi64 X(params.n, params.d);
    vector2d_to_RowMatrixXi64(training_data, X);
    X *= SCALING_FACTOR;
    X /= 255;

    read_MNIST_labels<uint64_t>(true, training_labels);
    ColVectorXi64 Y(params.n);
    vector_to_ColVectorXi64(training_labels, Y);
    Y *= SCALING_FACTOR;
    Y /= 10;

    

    LinearRegression linear_regression(X, Y, params, io);

    cout << "=======" << endl;
    cout << "Testing" << endl;
    cout << "=======" << endl;

    vector<double> testing_labels;
    int n_;

    vector<vector<double>> testing_data;
    read_MNIST_data<double>(false, testing_data, n_, params.d);

    RowMatrixXd testX(n_, params.d);
    vector2d_to_RowMatrixXd(testing_data, testX);
    testX /= 255.0;
    read_MNIST_labels<double>(false, testing_labels);

    ColVectorXd testY(n_);
    vector_to_ColVectorXd(testing_labels, testY);
    linear_regression.test_model(testX, testY);
    return 0;
}

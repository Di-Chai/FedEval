#include "read_MNIST.hpp"
#include "generate_random.hpp"
#include "linear_regression.hpp"
#include "online_phase.hpp"
#include <ctime>
#include <sys/time.h>

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

    params.n = NUM_IMAGES;
    params.d = 1000;

    generate_random_x_y_uint(params.n, params.d, training_data, training_labels);

    cout << "Data Generated" << endl;
    
    RowMatrixXi64 X(params.n, params.d);
    vector2d_to_RowMatrixXi64(training_data, X);
    X *= SCALING_FACTOR;

    cout << "X Converted" << endl;

    ColVectorXi64 Y(params.n);
    vector_to_ColVectorXi64(training_labels, Y);
    Y *= SCALING_FACTOR;

    cout << "Y Converted" << endl;
    
    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    cout << "Going to LR" << endl;
    
    LinearRegression linear_regression(X, Y, params, io);

    gettimeofday(&t2, NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << "Training Cost (timeuse) " << timeuse << " Seconds"  << endl;

    cout << "=======" << endl;
    cout << "Testing" << endl;
    cout << "=======" << endl;

    RowMatrixXd testX(params.n, params.d);
    ColVectorXd testY(params.n);

    descale<RowMatrixXi64, RowMatrixXd>(X, testX);
    descale<ColVectorXi64, ColVectorXd>(Y, testY);

    linear_regression.test_model(testX, testY);
    
    return 0;
}

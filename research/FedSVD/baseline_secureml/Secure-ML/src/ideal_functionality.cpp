#include "read_MNIST.hpp"
#include "util.hpp"
#include <math.h>

using namespace Eigen;
using Eigen::Matrix;
using namespace std;

Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

int NUM_IMAGES = BATCH_SIZE;

struct TrainingParams{
    int n, d;
    double alpha = 1.0/LEARNING_RATE_INV;
};

class LinearRegression{
public:
    double alpha;
    int n, d, t;
    RowMatrixXd X;
    ColVectorXd Y;
    ColVectorXd w;
    LinearRegression(RowMatrixXd& training_data, ColVectorXd& training_labels,
                     TrainingParams params){
        this->n = params.n;
        this->d = params.d;
        this->t = (params.n)/BATCH_SIZE;
        this->alpha = params.alpha;
        X = training_data;
        Y = training_labels;
        w.resize(d);
        for(int i = 0; i < d; i++)
            w[i] = 0;
        train_model();
    }

    void train_batch(int iter, int indexLo){
        RowMatrixXd Xb = X.block(indexLo, 0, BATCH_SIZE, d);
        ColVectorXd Yb = Y.segment(indexLo, BATCH_SIZE);

        ColVectorXd Y_(BATCH_SIZE);
        ColVectorXd D(BATCH_SIZE);
        ColVectorXd delta(d);

        Y_ = Xb * w;

        D = Y_ - Yb;

        delta = Xb.transpose() * D;

        delta = (delta * alpha)/BATCH_SIZE;

        w -= delta;

        /*
        ColVectorXd cost = X*w - Y;
        cost *= cost;
        double cost_ = cost.mean();
        cout << "Cost: " << cost_ << endl;
        */
    }

    void train_model(){
        for (int i = 0; i < t; i++){
            int indexLo = (i * BATCH_SIZE) % n;
            train_batch(i, indexLo);
        }
    }

    void test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels){
        ColVectorXd prediction;
        prediction = testing_data * w;
        prediction *= 10;
        int n_ = testing_labels.rows();

        ColVectorXd error;
        prediction = round(prediction.array());

        int num_correct = 0;
        for (int i = 0; i < n_; i++){
            if(prediction[i] == testing_labels[i])
                num_correct++;
        }
        double accuracy = num_correct/((double) n_);
        cout << "Accuracy on testing the trained model is " << accuracy * 100 << endl;
    }
};

int main(int argc, char** argv){

    int num_iters = atoi(argv[1]);
    NUM_IMAGES *= num_iters;

    TrainingParams params;

    cout << "========" << endl;
    cout << "Training" << endl;
    cout << "========" << endl;

    vector<vector<double>> training_data;
    vector<double> training_labels;

    read_MNIST_data<double>(true, training_data, params.n, params.d);
    RowMatrixXd X(params.n, params.d);
    vector2d_to_RowMatrixXd(training_data, X);
    X /= 255.0;

    read_MNIST_labels<double>(true, training_labels);
    ColVectorXd Y(params.n);
    vector_to_ColVectorXd(training_labels, Y);
    Y /= 10.0;

    LinearRegression linear_regression(X, Y, params);

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

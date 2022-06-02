#include "read_MNIST.hpp"
#include "util.hpp"
#include "generate_random.hpp"
#include <math.h>
#include <cmath>

using namespace Eigen;
using Eigen::Matrix;
using namespace std;

Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

int NUM_IMAGES = BATCH_SIZE;

struct TrainingParams{
    int n, d;
    // double alpha = 1.0/LEARNING_RATE_INV;
    double alpha = 0.5;
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
        default_random_engine generator;
        normal_distribution<double> random_normal(0, 1.0);
        for(int i = 0; i < d; i++)
            w[i] = random_normal(generator);
        for(int i = 0; i < 10; i++)
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

    }

    void train_loss(){
        ColVectorXd cost = X*w - Y;
        double mse = 0;
        for(int i=0; i<X.rows(); i++)
            mse += pow(cost[i], 2);
        cout << "Cost: " << mse / X.rows() << endl;
    }

    void train_model(){
        for (int i = 0; i < t; i++){
            int indexLo = (i * BATCH_SIZE) % n;
            train_batch(i, indexLo);
        }
        train_loss();
    }

    void test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels){
        ColVectorXd prediction;
        prediction = testing_data * w;

        int n_ = testing_labels.rows();

        double mean_square_error = 0;
        for(int i = 0; i < n_; i++){
            mean_square_error += pow(prediction[i] - testing_labels[i], 2);
        }
        cout << "MSE on testing the trained model is " << mean_square_error / n_ << endl;
        cout << "RMSE on testing the trained model is " << sqrt(mean_square_error / n_) << endl;
    }
};

int main(int argc, char** argv){

    srand((int)time(0));

    int num_iters = atoi(argv[1]);
    NUM_IMAGES *= num_iters;

    TrainingParams params;
    
    cout << "========" << endl;
    cout << "Training" << endl;
    cout << "========" << endl;

    vector<vector<double>> training_data;
    vector<double> training_labels;

    params.n = NUM_IMAGES;
    params.d = 1000;

    generate_random_x_y(params.n, params.d, training_data, training_labels);

    RowMatrixXd X(params.n, params.d);
    vector2d_to_RowMatrixXd(training_data, X);
    
    ColVectorXd Y(params.n);
    vector_to_ColVectorXd(training_labels, Y);
    
    LinearRegression linear_regression(X, Y, params);

    cout << "=======" << endl;
    cout << "Testing" << endl;
    cout << "=======" << endl;
    
    linear_regression.test_model(X, Y);
    return 0;
}

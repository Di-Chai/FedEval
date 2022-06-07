#include "linear_regression.hpp"
#include <cmath>

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

void LinearRegression::train_model(){
    for (int i = 0; i < t; i++){
        int indexLo = (i * BATCH_SIZE) % n;
        online->train_batch(i, indexLo);
    }

    if (party == BOB){
        send<ColVectorXi64>(io, online->wi);
    }
    else
        recv<ColVectorXi64>(io, w);

    if (party == ALICE){
        send<ColVectorXi64>(io, online->wi);
    }
    else
        recv<ColVectorXi64>(io, w);

    w += online->wi;

    descale<ColVectorXi64, ColVectorXd>(w, w_d);
}

// void LinearRegression::test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels){
//     ColVectorXd prediction;
//     prediction = testing_data * w_d;
//     prediction *= 10;
//     int n_ = testing_labels.rows();

//     ColVectorXd error;
//     prediction = round(prediction.array());

//     int num_correct = 0;
//     for (int i = 0; i < n_; i++){
//         if(prediction[i] == testing_labels[i])
//             num_correct++;
//     }
//     double accuracy = num_correct/((double) n_);
//     cout << "Accuracy on testing the trained model is " << accuracy * 100 << endl;
// }


void LinearRegression::test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels){
        ColVectorXd prediction;
        prediction = testing_data * w_d;

        int n_ = testing_labels.rows();

        double mean_square_error = 0;
        for(int i = 0; i < n_; i++){
            mean_square_error += pow(prediction[i] - testing_labels[i], 2);
        }
        cout << "MSE on testing the trained model is " << mean_square_error / n_ << endl;
        cout << "RMSE on testing the trained model is " << sqrt(mean_square_error / n_) << endl;
}
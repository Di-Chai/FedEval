#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <ctime>
using namespace std;


void generate_random_x_y(int number_samples, int num_features, vector<vector<double> > &X, vector<double> &y){

    cout << "Synthetic " << number_samples << " Samples " << num_features << " Features" << endl;

    default_random_engine generator;
    normal_distribution<double> random_normal(0, 1.0);
    normal_distribution<double> noise(1, 1.0);

    // Random X
    for(int i=0; i<number_samples; i++)
    {
        vector<double> tmp;
        for(int j=0; j<num_features; j++){
            tmp.push_back(random_normal(generator));
        }
        X.push_back(tmp);
    }
    //Ground Truth
    double num_informative = (int) (0.8 * number_samples);
    vector<double> ground_truth;
    for(int i=0; i<num_features; i++){
        if(i < num_informative){
            ground_truth.push_back(random_normal(generator));
        }
        else{
            ground_truth.push_back(0);
        }
    }
    // Generate Y
    for(int i=0; i<number_samples; i++){
        double tmp = 0;
        for(int j=0; j<num_features; j++){
            tmp += (X[i][j] * ground_truth[j]);
        }
        y.push_back(tmp + noise(generator));
    }
}

void generate_random_x_y_uint(int number_samples, int num_features, vector<vector<uint64_t> > &X, vector<uint64_t> &y){

    cout << "Synthetic " << number_samples << " Samples " << num_features << " Features" << endl;

    // Random X
    for(int i=0; i<number_samples; i++)
    {
        vector<uint64_t> tmp;
        for(int j=0; j<num_features; j++){
            tmp.push_back(rand());
        }
        X.push_back(tmp);
    }
    //Ground Truth
    double num_informative = (int) (0.8 * number_samples);
    vector<double> ground_truth;
    for(int i=0; i<num_features; i++){
        if(i < num_informative){
            ground_truth.push_back(rand());
        }
        else{
            ground_truth.push_back(0);
        }
    }
    // Generate Y
    for(int i=0; i<number_samples; i++){
        double tmp = 0;
        for(int j=0; j<num_features; j++){
            tmp += (X[i][j] * ground_truth[j]);
        }
        y.push_back(tmp);
    }
}
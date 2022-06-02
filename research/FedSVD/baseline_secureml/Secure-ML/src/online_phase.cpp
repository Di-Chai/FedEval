#include "online_phase.hpp"

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

void OnlinePhase::initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi){
    this->Xi = Xi;
    this->Yi = Yi;

    for (int i = 0; i < d; i++){
        wi(i) = 0;
    }

    Ui = triples->Ai;

    Ei = Xi - Ui;

    Vi = triples->Bi;
    Vi_ = triples->Bi_;
    Zi = triples->Ci;
    Zi_ = triples->Ci_;

    if (party == ALICE)
        send<RowMatrixXi64>(io, Ei);
    else
        recv<RowMatrixXi64>(io, E);
    if (party == BOB)
        send<RowMatrixXi64>(io, Ei);
    else
        recv<RowMatrixXi64>(io, E);

    E += Ei;

    std::cout << "OnlinePhase::initialize done" << std::endl;
    
}

void OnlinePhase::train_batch(int iter, int indexLo){
    RowMatrixXi64 X = Xi.block(indexLo, 0, BATCH_SIZE, d);
    ColVectorXi64 Y = Yi.segment(indexLo, BATCH_SIZE);
    RowMatrixXi64 Eb = E.block(indexLo, 0, BATCH_SIZE, d);
    ColVectorXi64 V = Vi.col(iter);
    ColVectorXi64 V_ = Vi_.col(iter);
    ColVectorXi64 Z = Zi.col(iter);
    ColVectorXi64 Z_ = Zi_.col(iter);

    Fi = wi - V;

    ColVectorXi64 D(BATCH_SIZE);
    ColVectorXi64 Y_(BATCH_SIZE);
    ColVectorXi64 Fi_(BATCH_SIZE);
    ColVectorXi64 F_(BATCH_SIZE);
    ColVectorXi64 delta(d);

    if (party == ALICE)
        send<ColVectorXi64>(io, Fi);
    else
        recv<ColVectorXi64>(io, F);

    if (party == BOB)
        send<ColVectorXi64>(io, Fi);
    else
        recv<ColVectorXi64>(io, F);

    F += Fi;

    Y_ = -i*(Eb * F)  + X * F + Eb * wi + Z;

    truncate<ColVectorXi64>(i, SCALING_FACTOR, Y_);

    D = Y_ - Y;

    Fi_ = D - V_;

    if (party == ALICE)
        send<ColVectorXi64>(io, Fi_);
    else
        recv<ColVectorXi64>(io, F_);

    if (party == BOB)
        send<ColVectorXi64>(io, Fi_);
    else
        recv<ColVectorXi64>(io, F_);

    F_ += Fi_;

    RowMatrixXi64 Et= Eb.transpose();
    RowMatrixXi64 Xt= X.transpose();

    delta = -i*(Et * F_) + Xt * F_ + Et * D + Z_;

    truncate<ColVectorXi64>(i, SCALING_FACTOR, delta);
    truncate<ColVectorXi64>(i, alpha_inv * BATCH_SIZE, delta);

    wi -= delta;
}

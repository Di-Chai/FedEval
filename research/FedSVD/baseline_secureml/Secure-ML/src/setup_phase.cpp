#include "setup_phase.hpp"
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <unistd.h>

using namespace Eigen;
using Eigen::Matrix;
using namespace emp;
using namespace std;

#define VMRSS_LINE 22

// get current process pid
int GetCurrentPid()
{
    return getpid();
}

float GetMemoryUsage(int pid)
{
    char file_name[64] = { 0 };
    FILE* fd;
    char line_buff[512] = { 0 };
    sprintf(file_name, "/proc/%d/status", pid);

    fd = fopen(file_name, "r");
    if (nullptr == fd)
        return 0;

    char name[64];
    int vmrss = 0;
    for (int i = 0; i < VMRSS_LINE - 1; i++)
        fgets(line_buff, sizeof(line_buff), fd);

    fgets(line_buff, sizeof(line_buff), fd);
    sscanf(line_buff, "%s %d", name, &vmrss);
    fclose(fd);

    // cnvert VmRSS from KB to MB
    return vmrss / 1048576.0;
}

void SetupPhase::initialize_matrices(){
    int current_pid = GetCurrentPid();
    prg.random_data(Ai.data(), (long) n * d * 8);
    prg.random_data(Bi.data(), d * t * 8);
    prg.random_data(Bi_.data(), BATCH_SIZE * t * 8);
}

void SetupPhase::generateMTs(){
    int current_pid = GetCurrentPid();
    vector<vector<uint64_t>> ci(t, vector<uint64_t>(BATCH_SIZE));
    vector<vector<uint64_t>> ci_(t, vector<uint64_t>(d));
    for(int i = 0; i < t; i++){
//         if (party == ALICE)
//             cout << "Debug iteration " << i << " 0 " << GetMemoryUsage(current_pid) << endl;
        RowMatrixXi64 Ai_b = Ai.block(i * BATCH_SIZE, 0, BATCH_SIZE, d);
        vector<vector<uint64_t>> ai(BATCH_SIZE, vector<uint64_t>(d));
        RowMatrixXi64_to_vector2d(Ai_b, ai);
        RowMatrixXi64 Ai_bt = Ai_b.transpose();
        vector<vector<uint64_t>> ai_t(d, vector<uint64_t>(BATCH_SIZE));
        RowMatrixXi64_to_vector2d(Ai_bt, ai_t);
        vector<uint64_t> bi = ColVectorXi64_to_vector(Bi.col(i));
        vector<uint64_t> bi_ = ColVectorXi64_to_vector(Bi_.col(i));
//         if (party == ALICE)
//             cout << "Debug iteration " << i << " 1 " << GetMemoryUsage(current_pid) << endl;
        secure_mult(BATCH_SIZE, d, ai, bi, ci[i]);
//         if (party == ALICE)
//             cout << "Debug iteration " << i << " 2 " << GetMemoryUsage(current_pid) << endl;
        secure_mult(d, BATCH_SIZE, ai_t, bi_, ci_[i]);
    }
    vector2d_to_ColMatrixXi64(ci, Ci);
    vector2d_to_ColMatrixXi64(ci_, Ci_);
    cout << "Triples Generated " << GetMemoryUsage(current_pid) << endl;
#if DEBUG
    verify();
#endif
}

void SetupPhase::secure_mult(int N, int D, vector<vector<uint64_t>>& a,
                             vector<uint64_t>& b, vector<uint64_t> &c){
    int NUM_OT[BITLEN];
    int total_ot = 0, num_ot;
    int current_pid = GetCurrentPid();

    // Calculate total number of OT
    for (int p = 0; p < 64; p++){
        int temp = 128/(64-p);
        NUM_OT[p] = N/temp;
        if (N % temp)
            NUM_OT[p]++;
        total_ot += NUM_OT[p];
    }
    total_ot *= D;


    block *x0, *x1, *rec;
    x0 = new block[total_ot];
    x1 = new block[total_ot];
    rec = new block[total_ot];

    int indexX0 = 0;
    int indexX1 = 0;

    bool bits_B[64];
    bool* sigma;
    sigma = new bool[total_ot];
    int index_sigma = 0;

//     if (party == ALICE) cout << "Debug secure_mult " << " 0 " << GetMemoryUsage(current_pid) << endl;

    uint64_t ***X0;
    X0 = new uint64_t**[N];
    for(int p = 0; p < N; p++) {
        X0[p] = new uint64_t*[BITLEN];
        for(int e = 0; e < BITLEN; e++){
            X0[p][e] = new uint64_t[D];
            prg.random_data(X0[p][e], D * 8);
        }
    }

//     if (party == ALICE) cout << "Debug secure_mult " << " 1 " << GetMemoryUsage(current_pid) << endl;

    for(int j = 0; j < D; j++){
        int_to_bool(bits_B, b[j]);

        for (int z = 0; z < 64; z++){
            num_ot = NUM_OT[z];
            uint64_t randomA[N];

            for (int p = 0; p < N; p++){
                randomA[p] = X0[p][z][j] + a[p][j];
            }

            int elements_in_block = 128/(64-z);
            int indexA = 0;

            for (int y = 0; y < num_ot; y++){
                sigma[index_sigma++] = bits_B[z];
            }

            for(int y = 0; y < num_ot; y++){
                int flag = elements_in_block;
                uint64_t temp_lo=0, temp_hi=0;
                uint64_t r_temp_lo=0, r_temp_hi=0;
                int elements_in_temp = 64/(64-z);
                int left_bitsLo = (64 % ((64-z)*elements_in_temp));

                r_temp_lo = (X0[indexA][z][j] << z);
                r_temp_lo >>= z;
                temp_lo = (randomA[indexA++] << z);
                temp_lo >>= z;
                flag--;
                for (int p=1; p<elements_in_temp; p++){
                    if (indexA <= N-1 && flag){
                        uint64_t r_next_element = (X0[indexA][z][j] << z);
                        r_next_element >>= z;
                        r_next_element <<= ((64-z) * p);
                        r_temp_lo ^= r_next_element;
                        uint64_t next_element = (randomA[indexA++] << z);
                        next_element >>= z;
                        next_element <<= ((64-z) * p);
                        temp_lo ^= next_element;
                        flag--;
                    }
                    else
                        break;
                }

                if (left_bitsLo){
                    if (indexA <= N-1 && flag){
                        uint64_t r_split_element = (X0[indexA][z][j] << z);
                        r_split_element >>= z;
                        r_temp_lo ^= (r_split_element << (64-left_bitsLo));
                        r_temp_hi ^= (r_split_element >> left_bitsLo);
                        uint64_t split_element = (randomA[indexA++] << z);
                        split_element >>= z;
                        temp_lo ^= (split_element << (64-left_bitsLo));
                        temp_hi ^= (split_element >> left_bitsLo);
                        flag--;
                    }
                }

                for (int p=0; p<elements_in_temp; p++){
                    if (indexA <= N-1 && flag){
                        uint64_t r_next_element = (X0[indexA][z][j] << z);
                        r_next_element >>= z;
                        if (left_bitsLo)
                            r_next_element <<= (((64-z)*p)+(64-z-left_bitsLo));
                        else
                            r_next_element <<= ((64-z)*p);
                        r_temp_hi ^= r_next_element;
                        uint64_t next_element = (randomA[indexA++] << z);
                        next_element >>= z;
                        if (left_bitsLo)
                            next_element <<= (((64-z)*p)+(64-z-left_bitsLo));
                        else
                            next_element <<= ((64-z)*p);
                        temp_hi ^= next_element;
                        flag--;
                    }
                    else
                        break;
                }

                x0[indexX0++] = makeBlock(r_temp_hi, r_temp_lo);
                x1[indexX1++] = makeBlock(temp_hi, temp_lo);

            }
        }
    }

//     if (party == ALICE) cout << "Debug secure_mult " << " 2 " << GetMemoryUsage(current_pid) << endl;
    
    if (party == ALICE){
        send_ot->send(x0, x1, total_ot);
    }
    else if (party == BOB){
        recv_ot->recv(rec, sigma, total_ot);
    }

//     if (party == ALICE) cout << "Debug secure_mult " << " 3 " << GetMemoryUsage(current_pid) << endl;

    if (party == BOB){
        send_ot->send(x0, x1, total_ot);
    }
    else if (party == ALICE){
        recv_ot->recv(rec, sigma, total_ot);
    }

//     if (party == ALICE) cout << "Debug secure_mult " << " 4 " << GetMemoryUsage(current_pid) << endl;

    int indexRec = 0;
    for (int j = 0; j < D; j++){
        for (int z = 0; z < 64; z++){
            int indexA = 0;
            num_ot = NUM_OT[z];
            int elements_in_block = 128/(64-z);

            for (int y = 0; y < num_ot; y++){
                int flag = elements_in_block;
                uint64_t temp_lo = extract_lo64(rec[indexRec]);
                uint64_t temp_hi = extract_hi64(rec[indexRec++]);

                int elements_in_temp = 64/(64-z);
                int left_bitsLo = (64 % ((64-z) * elements_in_temp));
                uint64_t mask;
                if((64 - z) < 64)
                    mask = ((1ULL << (64-z)) - 1);
                else
                    mask = -1;

                for(int p = 0; p < elements_in_temp; p++){
                    if (indexA <= N-1 && flag) {
                        uint64_t next_element = (temp_lo & mask);
                        next_element <<= z;
                        c[indexA++] += next_element;
                        temp_lo >>= 64-z;
                        flag--;

                    }
                    else
                        break;
                }
                if (left_bitsLo){
                    if (indexA <= N-1 && flag){
                        uint64_t split_mask;
                        if((64-z-left_bitsLo) < 64)
                            split_mask = ((1ULL << (64-z-left_bitsLo)) -1);
                        else
                            split_mask = -1;
                        uint64_t next_element = temp_hi & split_mask;
                        next_element <<= left_bitsLo;
                        next_element ^= temp_lo;
                        next_element <<= z;
                        c[indexA++] += next_element;
                        temp_hi >>= (64-z-left_bitsLo);
                        flag--;
                    }
                }
                for(int p = 0; p < elements_in_temp; p++){
                    if (indexA <= N-1 && flag) {
                        uint64_t next_element = (temp_hi & mask);
                        next_element <<= z;
                        c[indexA++] += next_element;
                        temp_hi >>= 64-z;

                        flag--;
                    }
                    else
                        break;
                }
            }
            for (int p = 0; p < N; p++){
                c[p] -= (X0[p][z][j] << z);
            }
        }
    }

//     if (party == ALICE) cout << "Debug secure_mult " << " 5 " << GetMemoryUsage(current_pid) << endl;

    for(int p = 0; p < N; p++) {
        for(int e = 0; e < BITLEN; e++){
            delete [] X0[p][e];
        }
        delete [] X0[p];
    }
    delete [] X0;
    delete [] x0;
    delete [] x1;
    delete [] rec;
    delete [] sigma;

//     if (party == ALICE) cout << "Debug secure_mult " << " 6 " << GetMemoryUsage(current_pid) << endl;

    for(int i = 0; i < N; i++){
        for(int k = 0; k < D; k++){
            c[i] += (a[i][k] * b[k]);
        }
    }
}

void SetupPhase::getMTs(SetupTriples *triples){
    triples->Ai = this->Ai;
    triples->Bi = this->Bi;
    triples->Ci = this->Ci;
    triples->Bi_ = this->Bi_;
    triples->Ci_ = this->Ci_;
}

void SetupPhase::verify(){
    if (party == ALICE) {
        RowMatrixXi64 Ai_b(BATCH_SIZE, d);
        ColMatrixXi64 Bi_b(d, 1); 
        ColMatrixXi64 Ci_b(BATCH_SIZE, 1);
        ColMatrixXi64 Bi_b_(BATCH_SIZE, 1);
        ColMatrixXi64 Ci_b_(d, 1);
        for(int i = 0; i < t; i++) {
            Ai_b = Ai.block((i * BATCH_SIZE) % n, 0, BATCH_SIZE, d);
            Bi_b = Bi.col(i);
            Ci_b = Ci.col(i);
            Bi_b_ = Bi_.col(i);
            Ci_b_ = Ci_.col(i);
            send(io, Ai_b);
            send(io, Bi_b);
            send(io, Ci_b);
            send(io, Bi_b_);
            send(io, Ci_b_);
        }
    }

    else {
        bool flag = true;
        bool flag_ = true;
        RowMatrixXi64 A(BATCH_SIZE, d);
        ColMatrixXi64 B(d, 1);
        ColMatrixXi64 C(BATCH_SIZE, 1);
        ColMatrixXi64 AB(BATCH_SIZE, 1);
        RowMatrixXi64 A_t(d, BATCH_SIZE);
        ColMatrixXi64 B_(BATCH_SIZE, 1);
        ColMatrixXi64 C_(d, 1);
        ColMatrixXi64 AB_(d, 1);
        for(int i = 0; i < t; i++) {
            RowMatrixXi64 Ai_b = Ai.block((i * BATCH_SIZE) % n, 0, BATCH_SIZE, d);
            ColMatrixXi64 Bi_b = Bi.col(i);
            ColMatrixXi64 Ci_b = Ci.col(i);
            ColMatrixXi64 Bi_b_ = Bi_.col(i);
            ColMatrixXi64 Ci_b_ = Ci_.col(i);

            recv(io, A);
            recv(io, B);
            recv(io, C);
            recv(io, B_);
            recv(io, C_);

            A += Ai_b;
            A_t = A.transpose();
            B += Bi_b;
            C += Ci_b;
            B_ += Bi_b_;
            C_ += Ci_b_;

            AB = A * B;
            AB_ = A_t * B_;

            if (C != AB) {
                flag_ = false;
                break;
            }

            if (C_ != AB_) {
                flag_ = false;
                break;
            }
        }

        if(flag == true) {
            cout << "Verification Successful" << endl;
        } else {
            cout << "Verification Failed" << endl;
        }

        if(flag_ == true) {
            cout << "Verification Successful" << endl;
        } else {
            cout << "Verification Failed" << endl;
        }
    }
}

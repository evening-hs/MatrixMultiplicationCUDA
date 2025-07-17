#ifndef MATRIX_CUH
#define MATRIX_CUH
#include <cuda_fp16.h>
#include <vector>
#include <string>

using namespace std;

class Matrix {
public:
    vector<half> data;
    int rows;
    int cols;
    Matrix(int rows, int cols);
    explicit Matrix(const string &filename);
    ~Matrix();
};


#endif //MATRIX_CUH
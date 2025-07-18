//
// Created by huertasg on 7/17/25.
//

#ifndef CSRMATRIX_CUH
#define CSRMATRIX_CUH

#include <cuda_fp16.h>
#include <string>

#include "Matrix.cuh"

class CSRMatrix {
public:
    int *hdr = nullptr;
    int *idx = nullptr;
    half *data = nullptr;
    CSRMatrix(const Matrix &matrix);
    ~CSRMatrix();
};


#endif //CSRMATRIX_CUH
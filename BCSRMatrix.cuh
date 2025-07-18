//
// Created by huertasg on 7/17/25.
//

#ifndef BCSRMATRIX_CUH
#define BCSRMATRIX_CUH
#include "Matrix.cuh"


class BCSRMatrix {
public:
    int *hdr = nullptr;
    int *idx = nullptr;
    Matrix **data = nullptr;
    explicit BCSRMatrix(const Matrix &matrix);
    ~BCSRMatrix();
};


#endif //BCSRMATRIX_CUH
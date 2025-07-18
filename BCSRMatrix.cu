//
// Created by huertasg on 7/17/25.
//

#include "BCSRMatrix.cuh"

BCSRMatrix::BCSRMatrix(const Matrix &matrix) {
    // find dense 16x16 blocks
    hdr = static_cast<int *>(malloc((matrix.rows / 16 + 1) * sizeof(int)));
    hdr[0] = 0;

    for (int i = 0; i < matrix.rows; i += 16) {
        hdr[i / 16 + 1] = hdr[i];
        for (int j = 0; j < matrix.cols; j += 16) {
            if (matrix.data[i * matrix.rows + j])
            {
                hdr[i / 16 + 1]++;
            }
        }
    }

    idx = static_cast<int *>(malloc(hdr[matrix.rows] * sizeof(int)));
    data = static_cast<Matrix **>(malloc(hdr[matrix.rows] * sizeof(Matrix*)));

    int k = 0;
    for (int i = 0; i < matrix.rows; i += 16) {
        for (int j = 0; j < matrix.cols; j += 16) {
            if (matrix.data[i * matrix.rows + j]) {
                idx[k] = j / 16;
                // obtain fragment
                data[k] = new Matrix(16, 16);
                for (int x = 0; x < 16; x++) {
                    for (int y = 0; y < 16; y++) {
                        data[k]->data[x * 16 + y] = matrix.data[(i + x) * matrix.cols + j + y];
                    }
                }
                k ++;
            }
        }
    }
}

BCSRMatrix::~BCSRMatrix() {
    free(hdr);
    free(idx);
    free(data);
}

//
// Created by huertasg on 7/17/25.
//

#include "BCSRMatrix.cuh"

#include <assert.h>
#include <iostream>

BCSRMatrix::BCSRMatrix(const Matrix &matrix) {
    // find dense 16x16 blocks
    blockRows = matrix.rows / 16;
    hdr = static_cast<int *>(malloc((blockRows + 1) * sizeof(int)));
    hdr[0] = 0;

    for (int i = 0; i < matrix.rows; i += 16) {
        hdr[i / 16 + 1] = hdr[i / 16];
        for (int j = 0; j < matrix.cols; j += 16) {
            if (matrix.data[i * matrix.rows + j])
            {
                hdr[i / 16 + 1]++;
            }
        }
    }

    idx = static_cast<int *>(malloc(hdr[blockRows] * sizeof(int)));
    data = static_cast<Matrix **>(malloc(hdr[blockRows] * sizeof(Matrix*)));

    int k = 0;
    for (int i = 0; i < matrix.rows; i += 16) {
        for (int j = 0; j < matrix.cols; j += 16) {
            if (matrix.data[i * matrix.cols + j]) {
                idx[k] = j / 16;
                // obtain fragment
                data[k] = new Matrix(16, 16);
                for (int x = 0; x < 16; x++) {
                    for (int y = 0; y < 16; y++) {
                        data[k]->data[x * 16 + y] = matrix.data[(i + x) * matrix.cols + j + y];
                    }
                }
                data[k]->nonZeros = 16 * 16;
                nonZeros += 16 * 16;
                k ++;
            }
        }
    }

    //assert(nonZeros == matrix.nonZeros);
}

BCSRMatrix::~BCSRMatrix() {
    free(hdr);
    free(idx);
    free(data);
}

void BCSRMatrix::print() const {
    std::cout << "hdr:\n\t";
    for (int i = 0; i < blockRows; i++) {
        std::cout << hdr[i] << " ";
    }
    std::cout << "\nidx:\n\t";
    for (int i = 0; i < hdr[blockRows]; i++) {
        std::cout << idx[i] << " ";
    }
    std::cout << "\ndata:\n\t";
    for (int i = 0; i < hdr[blockRows]; i++) {
        std::cout << "=== Block " << i << " ===\n";
        for (int j = 0; j < 16; j++) {
            cout << '\t';
            for (int k = 0; k < 16; k++) {
                cout << __half2float(data[i]->data[j * 16 + k]) << " ";
            }
            cout << '\n';
        }
    }
}

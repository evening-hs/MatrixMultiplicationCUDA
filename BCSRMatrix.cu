//
// Created by huertasg on 7/17/25.
//

#include "BCSRMatrix.cuh"

#include <cassert>
#include <iostream>

#define ASSERT_CUDA_SUCCESS error = cudaGetLastError(); \
                            if (error != cudaSuccess) \
                            cout << "BCSRMatrix::copyToDevice CUDA " \
                            "error: " << cudaGetErrorString(error) << '\n'; \
                            assert(error == cudaSuccess);


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

void BCSRMatrix::copyToDevice(int **gpuHdr, int **gpuIdx, half ***gpuData)
const {
    cudaError error;

    cudaMalloc(reinterpret_cast<void **>(gpuHdr),
               (blockRows + 1) * sizeof(int));
    ASSERT_CUDA_SUCCESS;

    cudaMalloc(reinterpret_cast<void **>(gpuIdx),
               hdr[blockRows] * sizeof(int));
    ASSERT_CUDA_SUCCESS;

    cudaMalloc(reinterpret_cast<void **>(gpuData),
               hdr[blockRows] * sizeof(half*));
    ASSERT_CUDA_SUCCESS;

    cudaMemcpy(*gpuHdr, hdr, (blockRows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    ASSERT_CUDA_SUCCESS;
    cudaMemcpy(*gpuIdx, idx, hdr[blockRows] * sizeof(int),
               cudaMemcpyHostToDevice);
    ASSERT_CUDA_SUCCESS;

    const auto cudaPtrs = static_cast<half **>(malloc(hdr[blockRows] * sizeof(half *)));

    for (int i = 0; i < hdr[blockRows]; i ++) {
        half *tmp;
        cudaMalloc(&tmp, 16 * 16 * sizeof(half));
        ASSERT_CUDA_SUCCESS;
        cudaMemcpy(tmp, data[i]->data, 16 * 16 * sizeof(half),
            cudaMemcpyHostToDevice);
        ASSERT_CUDA_SUCCESS;
        cudaPtrs[i] = tmp;
    }

    cudaMemcpy(*gpuData, cudaPtrs, hdr[blockRows] * sizeof(half *),
        cudaMemcpyHostToDevice);
    ASSERT_CUDA_SUCCESS;

    free(cudaPtrs);
}

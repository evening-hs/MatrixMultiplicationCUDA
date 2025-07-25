#include <cassert>
#include <iostream>
#include <chrono>
#include <mma.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "BCSRMatrix.cuh"
#include "CSRMatrix.cuh"
#include "Matrix.cuh"
#include "miscutil.h"

unsigned int N = 0;
constexpr unsigned int N_THREADS = 32;
string MATRIX_A_PATH = "../tests/MatrixA_4096_checkerboard.mat";
string MATRIX_B_PATH = "../tests/MatrixA_4096_random.mat";

using namespace std;
using namespace nvcuda;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define CHECK_CUDA_ERRORS \
    error = cudaGetLastError(); \
    if (error != cudaSuccess) \
        cout << "CUDA error: " << cudaGetErrorString(error) << '\n';
#define BYTES_SIZE(T) (N * N * sizeof(T))
#define MALLOC_MATRIX(T) static_cast<T *>(malloc(BYTES_SIZE(T)));
#define PREPARE_FUNC(_name) cout << "Running " << _name << "\n"; \
                            memset(memC, 0, BYTES_SIZE(float)); \
                            cudaMemset(gpuC, 0, BYTES_SIZE(float)); \
                            cudaEventCreate(&t1); \
                            cudaEventCreate(&t2); \
                            cudaEventRecord(t1, 0);
#define END_FUNC(_name, ...) cudaDeviceSynchronize(); \
                        error = cudaGetLastError(); \
                        if (error != cudaSuccess) \
                            cout << "CUDA error: " << cudaGetErrorString(error) << '\n'; \
                        cudaEventRecord(t2, 0); \
                        cudaEventSynchronize(t2); \
                        cudaEventElapsedTime(&ms, t1, t2); \
                        __VA_ARGS__ \
                        cudaMemcpy(memC, gpuC, BYTES_SIZE(float), cudaMemcpyDeviceToHost); \
                        cudaEventDestroy(t1); \
                        cudaEventDestroy(t2); \
                        printf("%30s time (ms): %10f\n", _name, ms); \
                        printf("%35s rmse: %10lf\n", _name, rmse(memC, correctMatrix, N)); \
                        printf("%31s max diff: %10lf\n", _name, maxdiff(memC, correctMatrix, N)); \
                        printf("%17s average relative error: %10lf\n", _name, avgrelerr(memC, correctMatrix, N));

/**
 * Dense matrix multiplication in CPU
 */
float *matrixMulCPU(const half *A, const half *B, float *C) {
    memset(C, 0, sizeof(float) * N * N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += __half2float(A[i * N + k]) * __half2float(
                    B[k * N + j]);
            }
        }
    }
    return C;
}

// MATRIX MULTIPLICATION ALGORITHMS

/**
 * Dense matrix multiplication in GPU
 * // O(n) per thread
 */
__global__ void denseMatrixMul(const half *d_A, const half *d_B, float *d_C,
                               const unsigned int n) {
    const unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        for (int k = 0; k < n; k++) {
            // Accumulate results for a single element
            // There's no need here to use reduction  or atomic add, because this
            // thread is the only one accessing this location
            d_C[rowIdx * n + colIdx] += __half2float(d_A[rowIdx * n + k]) *
                    __half2float(d_B[k * n + colIdx]);
        }
    }
}

/**
 * Multiply two dense matrices using tensors wmma
 */

__global__ void denseMatrixMulTensor(const half *d_A, const half *d_B,
                                     float *d_C, const unsigned int n) {
    // Calculate which 16x16 tile this thread block handles
    const unsigned int warp_row = blockIdx.y * 16;
    const unsigned int warp_col = blockIdx.x * 16;

    if (warp_row >= n || warp_col >= n) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Accumulate over K dimension in 16x16 chunks
    for (int k = 0; k < n; k += 16) {
        wmma::load_matrix_sync(a_frag, d_A + warp_row * n + k, n);
        wmma::load_matrix_sync(b_frag, d_B + k * n + warp_col, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(d_C + warp_row * n + warp_col, c_frag, n,
                            wmma::mem_row_major);
}


/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 *
 * O(R) R = non zeroes in this row
 */
__global__ void sparseMatrixMult1(const int *hdr, const int *idx,
                                  const half *data, const half *B, float *C,
                                  const unsigned int n) {
    const unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        for (int k = hdr[rowIdx]; k < hdr[rowIdx + 1]; k++) {
            C[rowIdx * n + colIdx] += __half2float(data[k]) * __half2float(
                B[idx[k] * n + colIdx]);
        }
    }
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 */
__global__ void sparseMatrixMult2(const int *hdr, const int *idx,
                                  const half *data, const half *B, float *C,
                                  const unsigned int n) {
    const unsigned int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < n) {
        int i = 0;
        for (int row = 0; row < n; row++) {
            for (; i < hdr[row + 1]; i++) {
                atomicAdd(&C[row * n + k],
                          __half2float(data[i]) * __half2float(
                              B[idx[i] * n + k]));
            }
        }
    }
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 */
__global__ void sparseMatrixMult3(const int *hdr, const int *idx,
                                  const half *data, const half *B, float *C,
                                  const unsigned int n) {
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < hdr[n]) {
        int row = 0;
        while (row < n && i >= hdr[row + 1]) row++;

        for (int k = 0; k < n; k++) {
            atomicAdd(&C[row * n + k],
                      __half2float(data[i]) * __half2float(B[idx[i] * n + k]));
        }
    }
}

/**
 * Multiply a BCSR matrix and a dense matrix using tensors
 */
__global__ void sparseMatrixMulTensor(const int *hdr, const int *idx,
                                      const half *data, const half *B,
                                      float *C, const unsigned int n) {
    const unsigned int warpRow = blockIdx.y * 16;
    const unsigned int warpCol = blockIdx.x * 16;

    if (warpRow >= n || warpCol >= n) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = hdr[warpRow / 16]; k < hdr[warpRow / 16 + 1]; k++) {
        wmma::load_matrix_sync(a_frag, data + k * 16 * 16, 16);
        wmma::load_matrix_sync(b_frag, B + idx[k] * 16 * n + warpCol, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpRow * n + warpCol, c_frag, n,
                            wmma::mem_row_major);
}

int main(const int argc, const char **argv) {
    if (argc == 3) {
        MATRIX_A_PATH = argv[1];
        MATRIX_B_PATH = argv[2];
    }

    cout << "Reading matrix A...\n";
    const Matrix *matrixA = new Matrix(MATRIX_A_PATH);
    cout << "Reading matrix B...\n";
    const Matrix *matrixB = new Matrix(MATRIX_B_PATH);
    assert(matrixA->cols == matrixB->rows);
    N = matrixA->cols;

    auto *memC = MALLOC_MATRIX(float);
    auto *correctMatrix = MALLOC_MATRIX(float);
    float *gpuC;
    half *gpuA_half, *gpuB_half, *gpuCSRData, *gpuBCSRData;
    int *gpuCSRHdr, *gpuCSRIdx, *gpuBCSRHdr, *gpuBCSRIdx;
    cudaEvent_t t1, t2;
    float ms = 0.0f;
    dim3 gridSize, blockSize;
    cudaError_t error;

    const auto *csrA = new CSRMatrix(*matrixA);
    const auto *bcsrA = new BCSRMatrix(*matrixA);
    bcsrA->copyToDevice(&gpuBCSRHdr, &gpuBCSRIdx, &gpuBCSRData);

    cudaMalloc(reinterpret_cast<void **>(&gpuA_half), BYTES_SIZE(half));
    cudaMalloc(reinterpret_cast<void **>(&gpuB_half), BYTES_SIZE(half));
    cudaMalloc(reinterpret_cast<void **>(&gpuC), BYTES_SIZE(float));
    cudaMalloc(reinterpret_cast<void **>(&gpuCSRData),
               csrA->hdr[N] * sizeof(half));
    cudaMalloc(reinterpret_cast<void **>(&gpuCSRHdr), (N + 1) * sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&gpuCSRIdx),
               csrA->hdr[N] * sizeof(int));

    cudaMemcpy(gpuA_half, matrixA->data, BYTES_SIZE(half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB_half, matrixB->data, BYTES_SIZE(half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpuCSRData, csrA->data, csrA->hdr[N] * sizeof(half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpuCSRHdr, csrA->hdr, (N + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpuCSRIdx, csrA->idx, csrA->hdr[N] * sizeof(int),
               cudaMemcpyHostToDevice);

    /* ========================== DENSE ON CPU ========================== */
#ifdef CHECK_CORRECTNESS
    PREPARE_FUNC("Dense on CPU");
    matrixMulCPU(matrixA->data, matrixB->data, correctMatrix);
    END_FUNC("Dense on CPU");
#endif

    /* ========================== DENSE ON GPU ========================== */
    gridSize = {
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        1
    };
    blockSize = {N_THREADS, N_THREADS, 1};
    PREPARE_FUNC("Dense on GPU");
    denseMatrixMul<<<gridSize, blockSize>>>(gpuA_half, gpuB_half, gpuC, N);
    END_FUNC("Dense on GPU");
    // Use dense on GPU as correct function
    memcpy(correctMatrix, memC, N * N * sizeof(float));

    /* ========================== DENSE WMMA ========================== */
    gridSize = {N / 16, N / 16, 1};
    blockSize = {32, 1, 1};
    PREPARE_FUNC("Dense WMMA");
    denseMatrixMulTensor<<<gridSize, blockSize>>
            >(gpuA_half, gpuB_half, gpuC, N);
    END_FUNC("Dense WMMA");

    /* ========================== SpMM 1 ========================== */
    gridSize = {
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        1
    };
    blockSize = {N_THREADS, N_THREADS, 1};
    PREPARE_FUNC("SpMM 1");
    sparseMatrixMult1<<<gridSize, blockSize>>>(gpuCSRHdr, gpuCSRIdx,
                                               gpuCSRData, gpuB_half, gpuC, N);
    END_FUNC("SpMM 1");

    /* ========================== SpMM 2 ========================== */
    gridSize = {
        N / (N_THREADS * N_THREADS) + (N % (N_THREADS * N_THREADS) > 0 ? 1 : 0),
        1, 1
    };
    blockSize = {N_THREADS * N_THREADS, 1, 1};
    PREPARE_FUNC("SpMM 2");
    sparseMatrixMult2<<<gridSize, blockSize>>>(gpuCSRHdr, gpuCSRIdx,
                                               gpuCSRData, gpuB_half, gpuC, N);
    END_FUNC("SpMM 2");

    /* ========================== SpMM 3 ========================== */
    gridSize = {
        csrA->hdr[N] / (N_THREADS * N_THREADS) + (
            csrA->hdr[N] % (N_THREADS * N_THREADS) > 0 ? 1 : 0),
        1,
        1
    };
    blockSize = {N_THREADS * N_THREADS, 1, 1};
    PREPARE_FUNC("SpMM 3");
    sparseMatrixMult3<<<gridSize, blockSize>>>(gpuCSRHdr, gpuCSRIdx,
                                               gpuCSRData, gpuB_half, gpuC, N);
    END_FUNC("SpMM 3");

    /* ========================= SpMM WITH TENSORS ========================= */
    gridSize = {N / 16, N / 16, 1};
    blockSize = {32, 1, 1};
    PREPARE_FUNC("SpMM with Tensors");
    sparseMatrixMulTensor<<<gridSize, blockSize>>>(gpuBCSRHdr, gpuBCSRIdx,
                                   gpuBCSRData, gpuB_half, gpuC, N);
    END_FUNC("SpMM with Tensors");

    /* ============================== CUBLAS =============================== */
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    constexpr float alpha = 1.0;
    constexpr float beta = 0.0;
    const int n = static_cast<int>(N);

    PREPARE_FUNC("cuBLAS GeMM");
    cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
        gpuB_half, CUDA_R_16F, n,
        gpuA_half, CUDA_R_16F, n,
        &beta, gpuC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT);
    END_FUNC("cuBLAS GeMM");

    cublasDestroy(cublasHandle);

    /* ============================= CUSPARSE ============================== */
    cusparseHandle_t cusparseHandle;
    size_t bufferSize;
    void *gpuBuffer = nullptr;
    cusparseMatDescr_t cusparseMatDescr;
    cusparseSpMatDescr_t matDescrA;
    cusparseDnMatDescr_t matDescrB, matDescrC;
    int64_t rows, cols, ld;
    cudaDataType_t dataType;
    cusparseOrder_t order;

    cusparseCreate(&cusparseHandle);

    cusparseCreateMatDescr(&cusparseMatDescr);
    cusparseSetMatType(cusparseMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(cusparseMatDescr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateCsr(&matDescrA, n, n, csrA->hdr[N],
        gpuCSRHdr, gpuCSRIdx, gpuCSRData,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    cusparseCreateDnMat(&matDescrB, n, n, n, gpuB_half,
        CUDA_R_16F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matDescrC, n, n, n, gpuC,
        CUDA_R_32F, CUSPARSE_ORDER_ROW);

    cusparseSpMM_bufferSize(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matDescrA, matDescrB,
        &beta, matDescrC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&gpuBuffer, bufferSize);

    PREPARE_FUNC("cuSPARSE CSR");
    cusparseSpMM(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matDescrA, matDescrB, &beta,
        matDescrC,CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, gpuBuffer);
    END_FUNC("cuSPARSE CSR",
        cusparseDnMatGet(matDescrC, &rows, &cols, &ld, reinterpret_cast<void **>(&gpuC), &dataType, &order););

    cusparseDestroySpMat(matDescrA);
    cusparseDestroyDnMat(matDescrB);
    cusparseDestroyDnMat(matDescrC);
    cusparseDestroy(cusparseHandle);

    free(memC);
    free(correctMatrix);
    cudaFree(gpuC);
    cudaFree(gpuA_half);
    cudaFree(gpuB_half);
    cudaFree(gpuCSRData);
    cudaFree(gpuCSRHdr);
    cudaFree(gpuCSRIdx);
    cudaFree(gpuBCSRData);
    cudaFree(gpuBCSRHdr);
    cudaFree(gpuBCSRIdx);

    return 0;
}

// vim: ts=4 sw=4

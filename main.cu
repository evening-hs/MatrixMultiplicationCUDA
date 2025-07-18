#include <cassert>
#include <iostream>
#include <chrono>
#include <mma.h>

#include "BCSRMatrix.cuh"
#include "CSRMatrix.cuh"
#include "Matrix.cuh"

unsigned int N = 0;
const unsigned int N_THREADS = 32;

using namespace std;
using namespace nvcuda;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define BYTES_SIZE(T) (N * N * sizeof(T))
#define MALLOC_MATRIX(T) static_cast<T *>(malloc(BYTES_SIZE(T)));
#define PREPARE_FUNC(_name) cout << "Running " << _name << "\n"; \
                            t1 = high_resolution_clock::now(); \
                            memset(memC, 0, BYTES_SIZE(float)); \
                            cudaMemset(gpuC, 0, BYTES_SIZE(float));
#define END_FUNC(_name) cudaDeviceSynchronize(); \
                        error = cudaGetLastError(); \
                        if (error != cudaSuccess) \
                            cout << "CUDA error: " << cudaGetErrorString(error) << '\n'; \
                        cudaMemcpy(memC, gpuC, BYTES_SIZE(float), cudaMemcpyDeviceToHost); \
                        t2 = high_resolution_clock::now(); \
                        ms = duration_cast<milliseconds>(t2 - t1); \
                        printf("%20s time (ms): %10d\n", _name, ms.count());

bool checkMatrix(const float *A, const float *B) {
#ifdef CHECK_CORRECTNESS

    for (int i = 0; i < N * N; i++)
        if (A[i] != B[i]) {
            std::cout << "Value at i = " << i << " mismatch " <<
                A[i] << "!=" << B[i] << '\n';
            return false;
        }
    std::cout << "Result is correct\n";
#endif
    return true;
}

/**
 * Dense matrix multiplication in CPU
 */
float *matrixMulCPU(const half *A, const half *B, float *C)
{
    memset(C, 0, sizeof(float) * N * N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[i * N + j] += __half2float(A[i * N + k]) * __half2float(B[k * N + j]);
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
__global__ void denseMatrixMul(const half *d_A, const half *d_B, float *d_C, const int n)
{
    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        for (int k = 0; k < n; k++)
        {
            // Accumulate results for a single element
            // There's no need here to use reduction  or atomic add, because this
            // thread is the only one accessing this location
            d_C[rowIdx * n + colIdx] += __half2float(d_A[rowIdx * n + k]) * __half2float(d_B[k * n + colIdx]);
        }
    }
}

/**
 * Multiply two dense matrices using tensors wmma
 */

__global__ void denseMatrixMulTensor(const half *d_A, const half *d_B, float *d_C, int n) {
    // Calculate which 16x16 tile this thread block handles
    int warp_row = blockIdx.y * 16;
    int warp_col = blockIdx.x * 16;

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

    wmma::store_matrix_sync(d_C + warp_row * n + warp_col, c_frag, n, wmma::mem_row_major);
}


/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 *
 * O(R) R = non zeroes in this row
 */
__global__ void sparceMatrixMult1(const int *hdr, const int *idx,
    const half *data, const half *B, float *C, const int n)
{
    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        for (int k = hdr[rowIdx]; k < hdr[rowIdx + 1]; k++) {
            C[rowIdx * n + colIdx] += __half2float(data[k]) * __half2float(B[idx[k] * n + colIdx]);
        }
    }
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 */
__global__ void sparceMatrixMult2(const int *hdr, const int *idx,
    const half *data, const half *B, float *C, const int n) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < n) {
        int i = 0;
        for (int row = 0; row < n; row++) {
            for (; i < hdr[row + 1]; i++) {
                atomicAdd(&C[row * n + k], __half2float(data[i]) * __half2float(B[idx[i] * n + k]));
            }
        }
    }
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 */
__global__ void sparceMatrixMult3(const int *hdr, const int *idx,
    const half *data, const half *B, float *C, const int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < hdr[n]) {
        int row = 0;
        while (row < n && i >= hdr[row + 1]) row ++;

        for (int k = 0; k < n; k++) {
            atomicAdd(&C[row * n + k], __half2float(data[i]) * __half2float(B[idx[i] * n + k]));
        }
    }
}

/**
 *
 *
 */
int main() {
    const Matrix *matrixA = new Matrix("../MatrixA.mat");
    const Matrix *matrixB = new Matrix("../MatrixB.mat");
    assert(matrixA->cols == matrixB->rows);
    N = matrixA->cols;
    const Matrix *matrixC = new Matrix(matrixA->rows, matrixA->cols);

    srand(time(nullptr));

    auto *memC = MALLOC_MATRIX(float);
    auto *correctMatrix = MALLOC_MATRIX(float);
    float *gpuC;
    half *gpuA_half, *gpuB_half, *gpuCSRData;
    int *gpuCSRHdr, *gpuCSRIdx;
    chrono::time_point<chrono::system_clock, chrono::system_clock::duration> t1, t2;
    duration<long, ratio<1, 1000>> ms;
    dim3 gridSize, blockSize;
    cudaError_t error;

    const auto *csrA = new CSRMatrix(*matrixA);
    const auto *bcsrA = new BCSRMatrix(*matrixA);

    cudaMalloc(reinterpret_cast<void **>(&gpuA_half), BYTES_SIZE(half));
    cudaMalloc(reinterpret_cast<void **>(&gpuB_half), BYTES_SIZE(half));
    cudaMalloc(reinterpret_cast<void **>(&gpuC), BYTES_SIZE(float));
    cudaMalloc(reinterpret_cast<void **>(&gpuCSRData), csrA->hdr[N] * sizeof(half));
    cudaMalloc(reinterpret_cast<void **>(&gpuCSRHdr), (N + 1) * sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&gpuCSRIdx), csrA->hdr[N] * sizeof(int));

    cudaMemcpy(gpuA_half, matrixA->data, BYTES_SIZE(half), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB_half, matrixB->data, BYTES_SIZE(half), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuCSRData, csrA->data, csrA->hdr[N] * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuCSRHdr, csrA->hdr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuCSRIdx, csrA->idx, csrA->hdr[N] * sizeof(int), cudaMemcpyHostToDevice);

#ifdef CHECK_CORRECTNESS
    PREPARE_FUNC("Matrix mult on CPU");
    matrixMulCPU(matrixA->data, matrixB->data, correctMatrix);
    END_FUNC("Matrix mult on CPU");
#endif

    gridSize = {
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        1};
    blockSize = {N_THREADS, N_THREADS, 1};
    PREPARE_FUNC("Matrix mult on GPU");
    denseMatrixMul<<<gridSize, blockSize>>>(gpuA_half, gpuB_half, gpuC, N);
    END_FUNC("Matrix mult on GPU");

    gridSize = {N/16, N/16, 1};
    blockSize = {32, 1, 1};
    PREPARE_FUNC("WMMA");
    denseMatrixMulTensor<<<gridSize, blockSize>>>(gpuA_half, gpuB_half, gpuC, N);
    END_FUNC("WMMA");
    checkMatrix(memC, correctMatrix);

    gridSize = {
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        N / N_THREADS + (N % N_THREADS > 0 ? 1 : 0),
        1};
    blockSize = {N_THREADS, N_THREADS, 1};
    PREPARE_FUNC("SpMM Algorithm 1");
    sparceMatrixMult1<<<gridSize, blockSize>>>(gpuCSRHdr, gpuCSRIdx,
        gpuCSRData, gpuB_half, gpuC, N);
    END_FUNC("SpMM Algorithm 1");
    checkMatrix(memC, correctMatrix);

    gridSize = {
        N / (N_THREADS * N_THREADS) + (N % (N_THREADS * N_THREADS) > 0 ? 1 : 0),
        1, 1};
    blockSize = {N_THREADS * N_THREADS, 1, 1};
    PREPARE_FUNC("SpMM Algorithm 2");
    sparceMatrixMult2<<<gridSize, blockSize>>>(gpuCSRHdr, gpuCSRIdx,
        gpuCSRData, gpuB_half, gpuC, N);
    END_FUNC("SpMM Algorithm 2");
    checkMatrix(memC, correctMatrix);

    gridSize = {
        csrA->hdr[N] / (N_THREADS * N_THREADS) + (csrA->hdr[N] % (N_THREADS * N_THREADS) > 0 ? 1 : 0),
        1,
        1
    };
    blockSize = {N_THREADS * N_THREADS, 1, 1};
    PREPARE_FUNC("SpMM Algorithm 3");
    sparceMatrixMult3<<<gridSize, blockSize>>>(gpuCSRHdr, gpuCSRIdx,
        gpuCSRData, gpuB_half, gpuC, N);
    END_FUNC("SpMM Algorithm 3");
    checkMatrix(memC, correctMatrix);

    free(memC);
    free(correctMatrix);
    cudaFree(gpuC);
    cudaFree(gpuA_half);
    cudaFree(gpuB_half);
    cudaFree(gpuCSRData);
    cudaFree(gpuCSRHdr);
    cudaFree(gpuCSRIdx);

    return 0;
}

// vim: ts=4 sw=4

#include <cassert>
#include <iostream>
#include <chrono>
#include <mma.h>
#include <vector>

const int N = 0x1000;// should be a multiple of 32
const int N_THREADS = 32;

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
                            cout << "CUDA error: %s\n" << cudaGetErrorString(error); \
                        cudaMemcpy(memC, gpuC, BYTES_SIZE(float), cudaMemcpyDeviceToHost); \
                        t2 = high_resolution_clock::now(); \
                        ms = duration_cast<milliseconds>(t2 - t1); \
                        cout << _name << " time (ms):\t" << ms.count() << endl; \

// CSR Matrix structure
struct CSRMatrix
{
    int *hdr = nullptr;
    int *idx = nullptr;
    float *data = nullptr;

    explicit CSRMatrix() {}

    explicit CSRMatrix(const float *M)
    {
        hdr = static_cast<int *>(malloc((N + 1) * sizeof(int)));
        hdr[0] = 0;

        for (int i = 0; i < N; i++)
        {
            hdr[i + 1] = hdr[i];
            for (int j = 0; j < N; j++)
            {
                if (M[i * N + j] != 0)
                {
                    hdr[i + 1]++;
                }
            }
        }

        idx = static_cast<int *>(malloc(hdr[N] * sizeof(int)));
        data = static_cast<float *>(malloc(hdr[N] * sizeof(float)));

        for (int i = 0, j = 0; i < N * N; i++)
        {
            if (M[i] != 0)
            {
                idx[j] = i % N;
                data[j] = M[i];
                j++;
            }
        }
    }

    void print() const
    {
        std::cout << "Header:\n";
        for (int i = 0; i < N + 1; i++)
        {
            std::cout << hdr[i] << ' ';
        }
        std::cout << "\nIndexes:\n";
        for (int i = 0; i < hdr[N]; i++)
        {
            std::cout << idx[i] << ' ';
        }
        std::cout << "\nData:\n";
        for (int i = 0; i < hdr[N]; i++)
        {
            std::cout << data[i] << ' ';
        }
        std::cout << '\n';
    }

    ~CSRMatrix()
    {
        free(hdr);
        free(idx);
        free(data);
    }
};

// UTILITIES

// generate random data for the matrix
void generateMatrix(float *M)
{
    for (int i = 0; i < N * N; i++)
    {
        M[i] = static_cast<float>(random() % 100);
    }
}

// generate a random sparce matrix with the specified sparsity percentage
void generateSparceMatrix(float *M, float sparsity)
{
    memset(M, 0.0f, N * N * sizeof(float));
    for (int i = 0; i < N * N; i++)
    {
        if (static_cast<float>(random()) / static_cast<float> (RAND_MAX) > sparsity)
        {
            M[i] = static_cast<float>(random() % 100);
        }
    }
}

// matrices are equal
bool equalMatrix(const float *A, const float *B) {
    for (int i = 0; i < N * N; i++)
        if (A[i] != B[i]) {
            std::cout << "Value at i = " << i << " mismatch " <<
                A[i] << "!=" << B[i] << '\n';
            return false;
        }
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
__global__ void denseMatrixMul(const float *d_A, const float *d_B, float *d_C, int n)
{

    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        for (int k = 0; k < n; k++)
        {
            // Accumulate results for a single element
            // There's no need here to use reduction  or atomic add, because this
            // thread is the only one accessing this location
            d_C[rowIdx * n + colIdx] += d_A[rowIdx * n + k] * d_B[k * n + colIdx];
        }
    }
}

/**
 * Multiply two dense matrices using tensors wmma
 */

__global__ void denseMatrixMulTensor(const half *d_A, const half *d_B, float *d_C) {
    // Calculate which 16x16 tile this thread block handles
    int warp_row = blockIdx.y * 16;
    int warp_col = blockIdx.x * 16;

    if (warp_row >= N || warp_col >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Accumulate over K dimension in 16x16 chunks
    for (int k = 0; k < N; k += 16) {
        wmma::load_matrix_sync(a_frag, d_A + warp_row * N + k, N);
        wmma::load_matrix_sync(b_frag, d_B + k * N + warp_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(d_C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
}


/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 *
 * O(R) R = non zeroes in this row
 */
__global__ void sparceMatrixMult1(const int *hdr, const int *idx,
    const float *data, const float *B, float *C, const int n)
{
    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (rowIdx < n && colIdx < n) {
        for (int k = hdr[rowIdx]; k < hdr[rowIdx + 1]; k++) {
            C[rowIdx * n + colIdx] += data[k] * B[idx[k] * n + colIdx];
        }
    }
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 */
__global__ void sparceMatrixMult2(const int *hdr, const int *idx,
    const float *data, const float *B, float *C, const int n) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < n) {
        int i = 0;
        for (int row = 0; row < n; row++) {
            for (; i < hdr[row + 1]; i++) {
                atomicAdd(&C[row * n + k], data[i] * B[idx[i] * n + k]);
            }
        }
    }
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 */
__global__ void sparceMatrixMult3(const int *hdr, const int *idx,
    const float *data, const float *B, float *C, const int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < hdr[n]) {
        int row = 0;
        while (row < n && i >= hdr[row + 1]) row ++;

        for (int k = 0; k < n; k++) {
            atomicAdd(&C[row * n + k], data[i] * B[idx[i] * n + k]);
        }
    }
}

int main() {
    srand(time(nullptr));

    float *memA = MALLOC_MATRIX(float);
    float *memB = MALLOC_MATRIX(float);
    float *memC = MALLOC_MATRIX(float);
    float *correctMatrix = MALLOC_MATRIX(float);
    half *memA_half = MALLOC_MATRIX(half);
    half *memB_half = MALLOC_MATRIX(half);
    float *gpuC;
    half *gpuA_half, *gpuB_half, *gpuCSRData;
    int *gpuCSRHdr, *gpuCSRIdx;
    chrono::time_point<chrono::system_clock, chrono::system_clock::duration> t1, t2;
    duration<long, ratio<1, 1000>> ms;
    dim3 gridSize, blockSize;
    cudaError_t error;

    generateSparceMatrix(memA, 0.0);
    generateMatrix(memB);
    CSRMatrix *csrA = new CSRMatrix(memA);

    for (int i = 0; i < N * N; i++) {
        memA_half[i] = __float2half(memA[i]);
        memB_half[i] = __float2half(memB[i]);
    }

    cudaMalloc(reinterpret_cast<void **>(&gpuC), BYTES_SIZE(float));
    cudaMalloc(reinterpret_cast<void **>(&gpuA_half), BYTES_SIZE(half));
    cudaMalloc(reinterpret_cast<void **>(&gpuB_half), BYTES_SIZE(half));
    cudaMalloc(reinterpret_cast<void **>(&gpuCSRData), csrA->hdr[N] * sizeof(half));
    cudaMalloc(reinterpret_cast<void **>(&gpuCSRHdr), (N + 1) * sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&gpuCSRIdx), csrA->hdr[N] * sizeof(int));

    cudaMemcpy(gpuC, memC, BYTES_SIZE(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuA_half, memA_half, BYTES_SIZE(half), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB_half, memB_half, BYTES_SIZE(half), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuCSRData, csrA->data, csrA->hdr[N] * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuCSRHdr, csrA->hdr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuCSRIdx, csrA->idx, csrA->hdr[N] * sizeof(int), cudaMemcpyHostToDevice);

#ifdef CHECK_CORRECTNESS
    PREPARE_FUNC("Matrix mult on CPU");
    matrixMulCPU(memA_half, memB_half, correctMatrix);
    END_FUNC("Matrix mult on CPU");
#endif

    gridSize = {N/16, N/16, 1};
    blockSize = {32, 1, 1};

    PREPARE_FUNC("WMMA");
    denseMatrixMulTensor<<<gridSize, blockSize>>>(gpuA_half, gpuB_half, gpuC);
    END_FUNC("WMMA");
#ifdef CHECK_CORRECTNESS
    equalMatrix(memC, correctMatrix);
#endif

    free(memA);
    free(memB);
    free(memC);
    free(correctMatrix);
    free(memA_half);
    free(memB_half);
    cudaFree(gpuC);
    cudaFree(gpuA_half);
    cudaFree(gpuB_half);
    cudaFree(gpuCSRData);
    cudaFree(gpuCSRHdr);
    cudaFree(gpuCSRIdx);

    return 0;
}

// vim: ts=4 sw=4

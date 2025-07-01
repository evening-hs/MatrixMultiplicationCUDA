#include <cassert>
#include <iostream>
#include <chrono>

int N = 0x4000; // should be a multiple of 32

using namespace std;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// CSR Matrix structure
struct CSRMatrix
{
    int *hdr = nullptr;
    int *idx = nullptr;
    float *data = nullptr;

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
float *matrixMulCPU(const float *A, const float *B, float *C)
{
    memset(C, 0, sizeof(float) * N * N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    return C;
}

// MATRIX MULTIPLICATION ALGORITHMS

/**
 * Dense matrix multiplication in GPU
 */
__global__ void denseMatrixMul(const float *d_A, const float *d_B, float *d_C, int n)
{

    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int k = 0; k < n; k++)
    {
        // Accumulate results for a single element
        // There's no need here to use reduction  or atomic add, because this
        // thread is the only one accessing this location
        d_C[rowIdx * n + colIdx] += d_A[rowIdx * n + k] * d_B[k * n + colIdx];
    }
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 */
__global__ void sparceMatrixMult1(const int *hdr, const int *idx,
    const float *data, const float *B, float *C, const int n)
{
    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int k = hdr[rowIdx]; k < hdr[rowIdx + 1]; k++)
    {
        C[rowIdx * n + colIdx] += data[k] * B[idx[k] * n + colIdx];
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
// sparce matrix multiplication

int main()
{
    srand(time(nullptr));
    // size of the matrices
    size_t bytes = N * N * sizeof(float);
    // allocate data for the host
    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C = nullptr;
    float *h_Correct = nullptr;
    h_A = static_cast<float *>(malloc(bytes));
    h_B = static_cast<float *>(malloc(bytes));
    h_C = static_cast<float *>(malloc(bytes));
    h_Correct = static_cast<float *>(malloc(bytes));

    memset(h_A, 0, bytes);

    // generate random matrix
    generateSparceMatrix(h_A, 0.80);
    generateMatrix(h_B);

    // parse matrix A to CSR format
    CSRMatrix csrA = CSRMatrix(h_A);

    // Allocate GPY memory
    int *d_hdr = nullptr, *d_idx = nullptr;
    float *d_data = nullptr, *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_hdr), (N + 1) * sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&d_idx), csrA.hdr[N] * sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&d_data), csrA.hdr[N] * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_A), bytes);
    cudaMalloc(reinterpret_cast<void **>(&d_B), bytes);
    cudaMalloc(reinterpret_cast<void **>(&d_C), bytes);

    // copy data from host to device
    cudaMemcpy(d_hdr, csrA.hdr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, csrA.idx, csrA.hdr[N] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, csrA.data, csrA.hdr[N] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Set result matrix to 0
    cudaMemset(d_C, 0, bytes);

    // define the grid size
    dim3 gridSize;
    dim3 blockSize;
    gridSize.x = N / 32;
    blockSize.x = 32;
    gridSize.y = N / 32;
    blockSize.y = 32;

    // clocks
    chrono::time_point<chrono::system_clock, chrono::system_clock::duration> t1, t2;
    duration<long, ratio<1, 1000>> ms;

    // RUN TESTS

#ifdef CHECK_CORRECTNESS
    // Compare results
    t1 = high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_Correct);
    t2 = high_resolution_clock::now();
    ms = duration_cast<chrono::milliseconds>(t2 - t1);
    cout << "matrixMulCPU time (ms):\t" << ms.count() << endl;
#endif

    // ### denseMatrixMul algorithm ###

    cudaMemset(d_C, 0, bytes);
    memset(h_C, 0, bytes);
    t1 = high_resolution_clock::now();
    denseMatrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    t2 = high_resolution_clock::now();
    ms = duration_cast<milliseconds>(t2 - t1);
    cout << "denseMatrixMul time (ms):\t" << ms.count() << endl;

#ifdef CHECK_CORRECTNESS
    if (equalMatrix(h_C, h_Correct)) {
        cout << "The result is correct." << endl;
    } else {
        cout << "The result is wrong." << endl;
    }
#endif

    // ### sparceMatrixMult1 algorithm ###

    cudaMemset(d_C, 0, bytes);
    memset(h_C, 0, bytes);
    t1 = high_resolution_clock::now();
    sparceMatrixMult1<<<gridSize, blockSize>>>(d_hdr, d_idx, d_data, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    t2 = high_resolution_clock::now();
    ms = duration_cast<chrono::milliseconds>(t2 - t1);
    cout << "sparceMatrixMult1 time (ms):\t" << ms.count() << endl;

#ifdef CHECK_CORRECTNESS
    if (equalMatrix(h_C, h_Correct)) {
        cout << "The result is correct." << endl;
    } else {
        cout << "The result is wrong." << endl;
    }
#endif

    // ### sparceMatrixMult2 algorithm ###
    // define the grid size
    gridSize.x = N / 32;
    blockSize.x = 32;
    gridSize.y = 1;
    blockSize.y = 1;
    cudaMemset(d_C, 0, bytes);
    memset(h_C, 0, bytes);

    t1 = high_resolution_clock::now();
    sparceMatrixMult2<<<gridSize, blockSize>>>(d_hdr, d_idx, d_data, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    t2 = high_resolution_clock::now();
    ms = duration_cast<chrono::milliseconds>(t2 - t1);
    cout << "sparceMatrixMult2 time (ms):\t" << ms.count() << endl;

#ifdef CHECK_CORRECTNESS
    if (equalMatrix(h_C, h_Correct)) {
        cout << "The result is correct." << endl;
    } else {
        cout << "The result is wrong." << endl;
    }
#endif

    // ### sparceMatrixMult3 algorithm ###
    // define the grid size
    gridSize.x = (csrA.hdr[N]) / 32 + (csrA.hdr[N] % 32 > 0 ? 1 : 0);
    blockSize.x = 32;
    gridSize.y = 1;
    blockSize.y = 1;
    cudaMemset(d_C, 0, bytes);
    memset(h_C, 0, bytes);

    t1 = high_resolution_clock::now();
    sparceMatrixMult3<<<gridSize, blockSize>>>(d_hdr, d_idx, d_data, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    t2 = high_resolution_clock::now();
    ms = duration_cast<chrono::milliseconds>(t2 - t1);
    cout << "sparceMatrixMult3 time (ms):\t" << ms.count() << endl;

#ifdef CHECK_CORRECTNESS
    if (equalMatrix(h_C, h_Correct)) {
        cout << "The result is correct." << endl;
    } else {
        cout << "The result is wrong." << endl;
    }
#endif

    // free the allocated ram
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_Correct);
    cudaFree(d_hdr);
    cudaFree(d_idx);
    cudaFree(d_data);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

// vim: ts=4 sw=4

#include <iostream>
#include <chrono>

int N = 10000;

using namespace std;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

struct CSRMatrix
{
    int *hdr = nullptr;
    int *idx = nullptr;
    float *data = nullptr;

    explicit CSRMatrix(const float *M)
    {
        hdr = (int *)malloc((N + 1) * sizeof(int));
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

// generate random data for the matrix
void generateMatrix(float *M)
{
    for (int i = 0; i < N * N; i++)
    {
        M[i] = static_cast<float>(random() % 100);
    }
}

// generate a random sparce matrix with the specified sparsity percentage
void generateSparceMatrix(float *M, int sparsity)
{
    memset(M, 0.0f, N * N * sizeof(float));
    for (int i = 0; i < N * N; i++)
    {
        if ((random() % 100) > sparsity)
        {
            M[i] = static_cast<float>(random() % 100);
        }
    }
}


/**
 * Dense matrix multiplication in GPU
 */
__global__ void matrixMul(const float *d_A, const float *d_B, float *d_C, int n)
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
__global__ void sparceMatrixMult(const int *hdr, const int *idx,
    const float *data, const float *B, float *C, const int n)
{
    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int k = hdr[rowIdx]; k < hdr[rowIdx + 1]; k++) // each col with non 0 in A
    {
        // I don't like this
        C[rowIdx * n + colIdx] += data[k] * B[idx[k] * n + colIdx];
    }
}

// sparce matrix multiplication

int main()
{
    // size of the matrixes
    size_t bytes = N * N * sizeof(float);
    // allocate data for the hosr
    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C = nullptr;
    h_A = static_cast<float *>(malloc(bytes));
    h_B = static_cast<float *>(malloc(bytes));
    h_C = static_cast<float *>(malloc(bytes));

    memset(h_A, 0, bytes);

    // generate random matrix
    generateSparceMatrix(h_A, 80);
    generateMatrix(h_B);

    // parse matrix A to CSR format
    CSRMatrix csrA = CSRMatrix(h_A);

    // Allocate GPY memory
    int *d_hdr = nullptr, *d_idx = nullptr;
    float *d_data = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_hdr), (N + 1) * sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&d_idx), csrA.hdr[N] * sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&d_data), csrA.hdr[N] * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_B), bytes);
    cudaMalloc(reinterpret_cast<void **>(&d_C), bytes);

    // copy data from host to device
    cudaMemcpy(d_hdr, csrA.hdr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, csrA.idx, csrA.hdr[N] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, csrA.data, csrA.hdr[N] * sizeof(float), cudaMemcpyHostToDevice);
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

    // run the code and calculate the execution time
    auto t1 = high_resolution_clock::now();
    //matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    sparceMatrixMult<<<gridSize, blockSize>>>(d_hdr, d_idx, d_data, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Wait for GPU to finish
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    auto t2 = high_resolution_clock::now();

    // calculate duration time of the serial code.
    auto ms_int = duration_cast<chrono::milliseconds>(t2 - t1);
    cout << "gpu : " << ms_int.count() << endl;

    // free the allocated ram
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_hdr);
    cudaFree(d_idx);
    cudaFree(d_data);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

// vim: ts=4 sw=4

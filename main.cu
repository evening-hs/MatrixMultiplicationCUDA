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

    explicit CSRMatrix(float *M)
    {
        hdr = (int *)malloc((N + 1) * sizeof(int));
        hdr[0] = 0;

        for (int i = 0; i < N; i++)
        {
            hdr[i + 1] = hdr[i];
            for (int j = 0; j < N; j++)
            {
                if (M[i * N + j])
                {
                    hdr[i + 1]++;
                }
            }
        }

        idx = (int *)malloc(hdr[N] * sizeof(int));
        data = (float *)malloc(hdr[N] * sizeof(float));

        for (int i = 0, j = 0; i < N * N; i++)
        {
            if (M[i])
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
        M[i] = rand() % 100;
    }
}

// generate a random sparce matrix with the specified sparcity percentage
void generateSparceMatrix(float *M, int sparcityPctg)
{
    for (int i = 0; i < N * N; i++)
    {
        if ((rand() % 100) > sparcityPctg)
        {
            M[i] = rand() % 100;
        }
    }
}

// this function runs at GPU and it multiply 2 matrixes.
__global__ void matrixMul(float *d_A, float *d_B, float *d_C, int n)
{

    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int k = 0; k < n; k++)
    {
        // Accumulate results for a single element
        d_C[rowIdx * n + colIdx] += d_A[rowIdx * n + k] * d_B[k * n + colIdx];
    }
}

/**
 * Multiply a CSR matrix x a dense matrix
 * C must be initialized and filled with 0s
 */
__global__ void *sparceMatrixMult(const CSRMatrix *A, const float *B, float *C, int n)
{
    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int k = A->hdr[rowIdx]; k < A->hdr[rowIdx + 1]; k++) // each col with non 0 in A
    {
        // I don't like this
        C[rowIdx * n + colIdx] += A->data[k] * B[A->idx[k] * n + colIdx];
    }
    return C;
}

// sparce matrix multiplication

int main()
{
    // size of the matrixes
    size_t bytes = N * N * sizeof(float);
    // allocate data for the hosr
    float *h_A;
    float *h_B;
    float *h_C;
    h_A = (float *)malloc(bytes);
    h_B = (float *)malloc(bytes);
    h_C = (float *)malloc(bytes);

    // generate random matrix
    generateSparceMatrix(h_A, 80);
    generateMatrix(h_B);

    // parse matrix A to CSR format
    CSRMatrix csrA = CSRMatrix(h_A);

    // allocate data at GPU ram
    float *d_B;
    float *d_C;
    CSRMatrix *d_A;
    cudaMalloc((void **)&d_A, sizeof(CSRMatrix));
    cudaMalloc((void **)&d_A->hdr, sizeof(int)*(N+1));
    cudaMalloc((void **)&d_A->idx, sizeof(int)*(csrA.hdr[N]));
    cudaMalloc((void **)&d_A->data, sizeof(float)*(csrA.hdr[N]));
    cudaMalloc((void **)&d_B, bytes);

    cudaError e = cudaMalloc((void **)&d_C, bytes);
    if (e != cudaSuccess)
    {
        printf("%s \n", cudaGetErrorString(e));
    }

    // copy data from RAM to GPU RAM
    cudaMemcpy(d_A->hdr, csrA.hdr, sizeof(int)*(N+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A->idx, csrA.idx, sizeof(int)*csrA.hdr[N], cudaMemcpyHostToDevice);
    cudaMemcpy(d_A->data, csrA.data, sizeof(float)*csrA.hdr[N], cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

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
    sparceMatrixMult<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    auto t2 = high_resolution_clock::now();

    // calculate duration time of the serial code.
    auto ms_int = duration_cast<chrono::milliseconds>(t2 - t1);
    cout << "gpu : " << ms_int.count() << endl;

    // free the allocated ram
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A->hdr);
    cudaFree(d_A->idx);
    cudaFree(d_A->data);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

// vim: ts=4 sw=4

/**
 * Compressed Sparce Row (CSR) matrix multiplication
 */

#include <iostream>

// Matrix size (N*N)
#define N 10

struct CSRMatrix
{
        int *hdr;
        int *idx;
        float *data;

        CSRMatrix(float *M)
        {
                hdr = new int[N+1];
                hdr[0] = 0;

                for (int i = 0; i < N; i++)
                {
                        hdr[i+1] = hdr[i];
                        for (int j = 0; j < N; j++)
                        {
                                if (M[i*N+j])
                                {
                                        hdr[i+1] ++;
                                }
                        }
                }
                
                idx = new int[hdr[N]];
                data = new float[hdr[N]];
                
                for (int i = 0, j = 0; i < N*N; i++)
                {
                        if (M[i])
                        {
                                idx[j] = i % N;
                                data[j] = M[i];
                                j ++;
                        }
                }
        }

        void print()
        {
                std::cout << "Header:\n";
                for (int i = 0; i < N+1; i++)
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
};

/**
 * generate a random sparce matrix with the specified sparcity percentage
 */
void generate_sparce_matrix(float *M, int sparcity_pctg) {
        for (int i = 0; i < N*N; i++) {
                if ((rand() % 100) > sparcity_pctg) {
                M[i] = rand() % 100;
                }
        }
}

int main(void)
{
        std::cout << "\n=== MATRIX A ===\n\n";
        float *M1 = new float[N*N];
        generate_sparce_matrix(M1, 80);

        for (int i = 0; i < N*N; i++)
        {
                std::cout << M1[i] << ' ';
                if ((i+1) % N == 0) std::cout << '\n';
        }

        CSRMatrix *A = new CSRMatrix(M1);
        A->print();

        std::cout << "\n=== MATRIX B ===\n\n";
        float *M2 = new float[N*N];
        generate_sparce_matrix(M2, 80);

        for (int i = 0; i < N*N; i++)
        {
                std::cout << M2[i] << ' ';
                if ((i+1) % N == 0) std::cout << '\n';
        }

        CSRMatrix *B = new CSRMatrix(M2);
        B->print();

        return 0;
}
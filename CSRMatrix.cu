#include "CSRMatrix.cuh"

CSRMatrix::CSRMatrix(const Matrix &matrix) {
    hdr = static_cast<int *>(malloc((matrix.rows + 1) * sizeof(int)));
    hdr[0] = 0;

    for (int i = 0; i < matrix.rows; i++)
    {
        hdr[i + 1] = hdr[i];
        for (int j = 0; j < matrix.cols; j++)
        {
            if (matrix.data[i * matrix.rows + j])
            {
                hdr[i + 1]++;
            }
        }
    }

    idx = static_cast<int *>(malloc(hdr[matrix.rows] * sizeof(int)));
    data = static_cast<half *>(malloc(hdr[matrix.rows] * sizeof(half)));

    for (int i = 0, j = 0; i < matrix.rows * matrix.cols; i++)
    {
        if (matrix.data[i])
        {
            idx[j] = i % matrix.rows;
            data[j] = matrix.data[i];
            j++;
        }
    }
}

CSRMatrix::~CSRMatrix() {
    free(hdr);
    free(idx);
    free(data);
}

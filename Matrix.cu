#include "Matrix.cuh"

#include <fstream>
#include <iostream>

Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->nonZeros = 0;
    this->data = static_cast<half *>(malloc((rows * cols) * sizeof(half)));
}

Matrix::Matrix(const string &filename) {
    this->rows = this->cols = 0;

    float v;
    freopen(filename.c_str(), "r", stdin);

    std::cin >> rows >> cols;
    this->data = static_cast<half *>(malloc((rows * cols) * sizeof(half)));

    for (int i = 0; i < rows * cols; i++) {
        std::cin >> v;
        if (v != 0.0f) this->nonZeros++;
        this->data[i] = __float2half(v);
    }
}

Matrix::~Matrix() {
    free(this->data);
};


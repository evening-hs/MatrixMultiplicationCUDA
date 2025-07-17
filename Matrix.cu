#include "Matrix.cuh"

#include <fstream>
#include <iostream>

Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
}

Matrix::Matrix(const string &filename) {
    this->rows = this->cols = 0;

    float v;
    freopen(filename.c_str(), "r", stdin);

    std::cin >> rows >> cols;
    data.resize(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        std::cin >> v;
        data[i] = __float2half(v);
    }
}

Matrix::~Matrix() = default;


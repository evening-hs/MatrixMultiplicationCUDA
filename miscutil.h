//
// Created by huertasg on 7/22/25.
//

#ifndef MISCUTIL_H
#define MISCUTIL_H

void printMatrix(const float *M, const unsigned int n);

bool checkMatrix(const float *A, const float *B, const unsigned int n);

double rmse(const float *A, const float *B, const unsigned int n);

float maxdiff(const float *A, const float *B, const unsigned int n);

float avgrelerr(const float *A, const float *B, const unsigned int n);

#endif //MISCUTIL_H

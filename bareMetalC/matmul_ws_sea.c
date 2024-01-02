#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "include/gemmini_testutils.h"

#define N 4 
#include "matrices_bf16_fp32_4.h"
#define MAT_DIM_I N
#define MAT_DIM_J N
#define MAT_DIM_K N
#define FULL_BIAS_WIDTH 0
static void printMatrixAccBig(acc_t m[N][N]) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j)
#ifndef ELEM_T_IS_FLOAT
      printf("%d ", m[i][j]);
#else
      printf("%x ", acc_t_to_acc_t_bits(m[i][j]));
#endif
    printf("\n");
  }
}
int main(){
    static elem_t A[N][N] row_align(1);
    static elem_t B[N][N]  row_align(1);


    #define TEST_NUM 10
    for (size_t test = 0; test < TEST_NUM; test++) {
        // Populate matrix from header file
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                A[i][j] = matrix_a[test][i][j];
                B[i][j] = matrix_b[test][i][j];
            }
        }
        static acc_t C[N][N] row_align(1);
        tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)A, (elem_t*)B, NULL , (acc_t*)C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            true, false,
            0,
            WS);
        printf("C:\n");
        printMatrixAccBig(C);
    }
    exit(0);
}

#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "ros_patch_match/convenience.cuh"

namespace Cuda {



__host__ __device__ float dist(const uchar *a,const uchar *b,const int a_cols, const int a_rows,
                               const int b_cols, const int b_rows, const int ax, const int ay,
                               const int bx, const int by,const int patch_w, float cutoff = INT_MAX);
void patchMatch(const uchar * a,const uchar * b,unsigned int * ann,unsigned int * newann,
                float * annd,float * newannd,
                const int cols, const int rows, const int iters,
                const int patch_size,const int rs_max);
}

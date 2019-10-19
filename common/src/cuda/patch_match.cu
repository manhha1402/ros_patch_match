#include "ros_patch_match/cuda_functions.cuh"
/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
namespace Cuda {

__device__ float cuRand(unsigned int * seed) {
    //random number in cuda
    unsigned long a = 16807;
    unsigned long m = 2147483647;
    unsigned long x = (unsigned long)* seed;
    x = (a*x) % m;
    *seed = (unsigned int)x;
    return ((float)x / m);
}


__host__ __device__ float dist(const uchar *a,const uchar *b,const int a_cols, const int a_rows,
                               const int b_cols, const int b_rows, const int ax, const int ay,
                               const int bx, const int by,const int HALF_PATCH, float cutoff)
{

    //suppose patch_w is an odd number
    float pixel_sum = 0, pixel_no = 0, pixel_dist=0;//number of pixels realy counted

    for (int dy = -HALF_PATCH; dy <= HALF_PATCH; dy++) {
        for (int dx = -HALF_PATCH; dx <= HALF_PATCH; dx++) {
            if (
                    (ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
                    &&
                    (by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
                    )//the pixel in a should exist and pixel in b should exist
            {
                int dr = a[(ay + dy) * 3 * a_cols + (ax + dx) * 3 + 2] - b[(by + dy) * 3 * b_cols + (bx + dx) * 3 + 2];
                int dg = a[(ay + dy) * 3 * a_cols + (ax + dx) * 3 + 1] - b[(by + dy) * 3 * b_cols + (bx + dx) * 3 + 1];
                int db = a[(ay + dy) * 3 * a_cols + (ax + dx) * 3 + 0] - b[(by + dy) * 3 * b_cols + (bx + dx) * 3 + 0];
                pixel_sum += (float)(dr*dr + dg*dg + db*db);
                pixel_no += 1;
            }
        }

    }
    pixel_dist = pixel_sum / pixel_no;
    if (pixel_dist >= cutoff) { return cutoff; }
    else {
        return pixel_dist;
    }

}
__device__ void improve_guess(const uchar * a,const uchar * b,
                              int a_cols, int a_rows,
                              int b_cols, int b_rows,
                              int ax, int ay, int &xbest,
                              int &ybest, float &dbest, int xp, int yp, int HALF_PATCH) {
    float d = 0;
    d = dist(a, b, a_cols, a_rows,b_cols,b_rows, ax, ay, xp, yp, HALF_PATCH, dbest);

    if (d < dbest) {
        xbest = xp;
        ybest = yp;
        dbest = d;
    }
}
__global__ void kernerPatchMatch(const uchar * a,const uchar * b,unsigned int * ann,float * annd,
                                 const int a_cols, const int a_rows,const int b_cols, const int b_rows,
                                 const int iters,const int patch_size,const int rs_max )

{
    int ax = blockIdx.x*blockDim.x + threadIdx.x;
    int ay = blockIdx.y*blockDim.y + threadIdx.y;
    if (ax < a_cols && ay < a_rows) {

        // for random number
        unsigned int seed = ay*a_cols + ax;

        for (int iter = 0; iter < iters; iter++) {

            /* Current (best) guess. */
            unsigned int v = ann[ay*a_cols + ax];
            int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
            float dbest = annd[ay*a_cols + ax];

            for (int jump = 8; jump > 0; jump /= 2) {

                //jump = jump * iterDirection;

                /* Propagation: Improve current guess by trying instead correspondences from left, right, up and downs. */
                if ((ax - jump) < a_cols&&(ax - jump) >= 0)//left
                {
                    unsigned int vp = ann[ay*a_cols + ax - jump];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp) + jump, yp = INT_TO_Y(vp);//the propagated match from vp
                    if (xp < b_cols && xp>=0)
                    {
                        //improve guress
                        improve_guess(a, b, a_cols, a_rows, b_cols, b_rows, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);

                    }
                }
                ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
                annd[ay*a_cols + ax] = dbest;


                if ((ax + jump) < a_cols)//right
                {
                    unsigned int vp = ann[ay*a_cols + ax + jump];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp) - jump, yp = INT_TO_Y(vp);
                    if (xp >= 0&&xp<b_cols)
                    {
                        //improve guress
                        improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);
                    }
                }

                ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
                annd[ay*a_cols + ax] = dbest;

                if ((ay - jump) < a_rows && (ay - jump) >=0)//up
                {
                    unsigned int vp = ann[(ay - jump)*a_cols + ax];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + jump;
                    if (yp >= 0 && yp <b_rows)
                    {
                        //improve guress
                        improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);
                    }
                }

                ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
                annd[ay*a_cols + ax] = dbest;

                if ((ay + jump) < a_rows)//down
                {
                    unsigned int vp = ann[(ay + jump)*a_cols + ax];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) - jump;
                    if (yp >= 0)
                    {
                        //improve guress
                        improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);
                    }
                }

                ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
                annd[ay*a_cols + ax] = dbest;
                __syncthreads();

            }

            /* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
            int rs_start = rs_max;
            if (rs_start > cuMax(b_cols, b_rows)) {
                rs_start = cuMax(b_cols, b_rows);
            }
            for (int mag = rs_start; mag >= 1; mag /= 2) {
                /* Sampling window */
                int xmin = cuMax(xbest - mag, 0), xmax = cuMin(xbest + mag + 1, b_cols);
                int ymin = cuMax(ybest - mag, 0), ymax = cuMin(ybest + mag + 1, b_rows);
                int xp = xmin + (int)(cuRand(&seed)*(xmax - xmin)) % (xmax - xmin);
                int yp = ymin + (int)(cuRand(&seed)*(ymax - ymin)) % (ymax - ymin);

                improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);

            }

            ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
            annd[ay*a_cols + ax] = dbest;
            __syncthreads();
        }

    }
}

void patchMatch(const uchar * a,const uchar * b,unsigned int * ann,unsigned int * newann,
                float * annd,float * newannd,
                const int a_cols, const int a_rows, const int b_cols,const int b_rows,
                const int iters, const int patch_size,const int rs_max)
{


    dim3 threads(32,32);
    dim3 blocks((cols + threads.x -1)/threads.x,
                (rows + threads.y -1)/threads.y);
    uchar *a_device, *b_device;
    unsigned int *ann_device;
    float * annd_device;

    cudaMalloc(&a_device, cols*rows*3* sizeof(uchar));
    cudaMalloc(&b_device, cols*rows*3 * sizeof(uchar));
    cudaMalloc(&annd_device, cols*rows * sizeof(float));
    cudaMalloc(&ann_device, cols*rows * sizeof(unsigned int));

    newann = (unsigned int *)malloc(cols*rows * sizeof(unsigned int));
    newannd = (float *)malloc(cols*rows * sizeof(float));

    cudaMemcpy(a_device, a, cols*rows*3 * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b, cols*rows*3 * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(ann_device, ann, cols*rows * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(annd_device, annd, cols*rows * sizeof(float), cudaMemcpyHostToDevice);
    kernerPatchMatch <<<blocks,threads>>> (a_device,b_device,ann_device,annd_device,cols,rows,iters,patch_size,rs_max);
    cudaSafeCall ( cudaGetLastError () );
    cudaMemcpy(newann, ann_device, cols*rows * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(newannd, annd_device, cols*rows * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(ann_device);
    cudaFree(annd_device);
}

}

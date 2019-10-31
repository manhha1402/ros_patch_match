#include "ros_patch_match/cuda_functions.cuh"
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
//L2 distance of RGB space between source patch and target patch

__host__ __device__ float distance(int * source, int * target, int s_rows, int s_cols, int t_rows, int t_cols,
                               int sx, int sy, int tx, int ty, int patch_size, float cutoff = INT_MAX) {
    // patch_size is an odd number
    float ans = 0, num = 0;//number of pixels realy counted
    for (int dy = -patch_size/2; dy <= patch_size/2; dy++) {
        for (int dx = -patch_size / 2; dx <= patch_size/2; dx++) {
            if (
                    (sy + dy) < s_rows && (sy + dy) >= 0 && (sx + dx) < s_cols && (sx + dx) >= 0
                    &&
                    (ty + dy) < t_rows && (ty + dy) >= 0 && (tx + dx) < t_cols && (tx + dx) >= 0
                    )//the pixel in a should exist and pixel in b should exist
            {
                int dr = source[(sy + dy) * 3 * s_cols + (sx + dx) * 3 + 2] - target[(ty + dy) * 3 * t_cols + (tx + dx) * 3 + 2];
                int dg = source[(sy + dy) * 3 * s_cols + (sx + dx) * 3 + 1] - target[(ty + dy) * 3 * t_cols + (tx + dx) * 3 + 1];
                int db = source[(sy + dy) * 3 * s_cols + (sx + dx) * 3 + 0] - target[(ty + dy) * 3 * t_cols + (tx + dx) * 3 + 0];
                ans += (float)(dr*dr + dg*dg + db*db);
                num += 1;
            }
        }

    }
    ans = ans / num;
    if (ans >= cutoff) { return cutoff; }
    else {
        return ans;
    }
}
//compare L2 distance of current best match and current match, replace current best match if distance is smaller

__device__ void compareDistance(int * source, int * target, int s_rows, int s_cols, int t_rows, int t_cols,
                              int sx, int sy, int &txbest, int &tybest, float &dbest, int xp, int yp, int patch_size) {
    float d = 0;
    d = distance(source, target, s_rows, s_cols, t_rows, t_cols, sx, sy, xp, yp, patch_size, dbest);

    if (d < dbest) {
        txbest = xp;
        tybest = yp;
        dbest = d;
    }
}

__global__ void kernelPatchMatch(int * source, int * target, unsigned int *ann, float *annd,
                                 const int s_rows,const int s_cols,const int t_rows, const int t_cols, const int iters,
                                 const int patch_size, const int threshold) {

    int sx = blockIdx.x*blockDim.x + threadIdx.x;
    int sy = blockIdx.y*blockDim.y + threadIdx.y;

    if (sx < s_cols&&sy < s_rows) {

        // for random number
        unsigned int seed = sy*s_cols + sx;

        for (int iter = 0; iter < iters; iter++) {


            /* Current (best) guess. */
            unsigned int v = ann[sy*s_cols + sx];
            int txbest = INT_TO_X(v), tybest = INT_TO_Y(v);
            float dbest = annd[sy*s_cols + sx];

            for (int jump = 8; jump > 0; jump /= 2) {


                /* Propagation: Improve current guess by trying instead correspondences from left, right, up and downs. */
                if ((sx - jump) < s_cols&&(sx - jump) >= 0)//left
                {
                    unsigned int vp = ann[sy*s_cols + sx - jump];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp) + jump, yp = INT_TO_Y(vp);//the propagated match from vp
                    if (xp < t_cols && xp>=0)
                    {

                        compareDistance(source, target, s_rows, s_cols, t_rows, t_cols, sx, sy, txbest, tybest, dbest, xp, yp, patch_size);

                    }
                }
                ann[sy*s_cols + sx] = XY_TO_INT(txbest, tybest);
                annd[sy*s_cols + sx] = dbest;


                if ((sx + jump) < s_cols)//right
                {
                    unsigned int vp = ann[sy*s_cols + sx + jump];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp) - jump, yp = INT_TO_Y(vp);
                    if (xp >= 0&&xp<t_cols)
                    {

                        compareDistance(source, target, s_rows, s_cols, t_rows, t_cols, sx, sy, txbest, tybest, dbest, xp, yp, patch_size);
                    }
                }

                ann[sy*s_cols + sx] = XY_TO_INT(txbest, tybest);
                annd[sy*s_cols + sx] = dbest;

                if ((sy - jump) < s_rows && (sy - jump) >=0)//up
                {
                    unsigned int vp = ann[(sy - jump)*s_cols + sx];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + jump;
                    if (yp >= 0 && yp <t_rows)
                    {

                        compareDistance(source, target, s_rows, s_cols, t_rows, t_cols, sx, sy, txbest, tybest, dbest, xp, yp, patch_size);
                    }
                }

                ann[sy*s_cols + sx] = XY_TO_INT(txbest, tybest);
                annd[sy*s_cols + sx] = dbest;

                if ((sy + jump) < s_rows)//down
                {
                    unsigned int vp = ann[(sy + jump)*s_cols + sx];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) - jump;
                    if (yp >= 0)
                    {

                        compareDistance(source, target, s_rows, s_cols, t_rows, t_cols, sx, sy, txbest, tybest, dbest, xp, yp, patch_size);
                    }
                }

                ann[sy*s_cols + sx] = XY_TO_INT(txbest, tybest);
                annd[sy*s_cols + sx] = dbest;
                __syncthreads();

            }

            /* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
            int rs_start = threshold;
            if (rs_start > cuMax(t_cols, t_rows)) {
                rs_start = cuMax(t_cols, t_rows);
            }
            for (int mag = rs_start; mag >= 1; mag /= 2) {
                /* Sampling window */
                int xmin = cuMax(txbest - mag, 0), xmax = cuMin(txbest + mag + 1, t_cols);
                int ymin = cuMax(tybest - mag, 0), ymax = cuMin(tybest + mag + 1, t_rows);
                int xp = xmin + (int)(cuRand(&seed)*(xmax - xmin)) % (xmax - xmin);
                int yp = ymin + (int)(cuRand(&seed)*(ymax - ymin)) % (ymax - ymin);

                compareDistance(source, target, s_rows, s_cols, t_rows, t_cols, sx, sy, txbest, tybest, dbest, xp, yp, patch_size);

            }

            ann[sy*s_cols + sx] = XY_TO_INT(txbest, tybest);
            annd[sy*s_cols + sx] = dbest;
            __syncthreads();
        }

    }
}



__host__ void convertcvMat2Array(const cv::Mat& img,int*& array)
{
    array = new int[img.rows * img.cols * 3];
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j <img.cols; j++)
        {
            cv::Vec3b rgb = img.at<cv::Vec3b>(i, j);
            array[i*img.cols * 3 + j * 3 + 0] = rgb.val[0];
            array[i*img.cols * 3 + j * 3 + 1] = rgb.val[1];
            array[i*img.cols * 3 + j * 3 + 2] = rgb.val[2];
        }
    }
}
__host__ void initAnn(int * source,int * target,unsigned int *& ann,
                float *& annd, int s_cols, int s_rows,
                int t_cols, int t_rows,int patch_size)
{
#pragma omp parallel for schedule(static)
    for (int sy = 0; sy < s_rows; sy++) {
        for (int sx = 0; sx < s_cols; sx++) {
            int tx = rand() % t_cols;
            int ty = rand() % t_rows;
            ann[sy*s_cols + sx] = XY_TO_INT(tx, ty);
            annd[sy*s_cols + sx] = distance(source,target,s_rows,s_cols,t_rows,
                                            t_cols,sx,sy,tx,ty,patch_size,INT_MAX);

        }
    }
}


void hostPatchMatch(const cv::Mat& source, const cv::Mat& target,const int iters, const int patch_size,
                    cv::Mat& cvMatann, cv::Mat& cvMatannd,cv::Mat& reconstructed_image)
{


    dim3 threads(32,32);
    dim3 blocks(source.cols/ threads.x + 1,source.rows/ threads.y + 1);
    int *source_host, *target_host,*source_device, *target_device;
    unsigned int *ann_host,*ann_device,*ann_final_host;
    float *annd_host,*annd_device,*annd_final_host;
    int *params_host, *params_device;


    const int size_of_source = source.cols * source.rows * 3;
    const int size_of_target = target.cols * target.rows * 3;
    const int size_of_ann = source.cols * source.rows;
    ///init host memory
    convertcvMat2Array(source,source_host);
    convertcvMat2Array(target,target_host);
    ann_host = (unsigned int*)malloc(size_of_ann*sizeof (unsigned int));
    annd_host = (float*)malloc(size_of_ann*sizeof (float));
    ann_final_host = (unsigned int*)malloc(size_of_ann*sizeof (unsigned int));
    annd_final_host = (float*)malloc(size_of_ann*sizeof (float));

    //initilize nearest neigbor
    initAnn(source_host,target_host,ann_host,annd_host,source.cols,source.rows,target.cols,target.rows,
            patch_size);


    ///init device memory
    cudaMalloc(&source_device, size_of_source * sizeof(int));
    cudaMalloc(&target_device, size_of_target * sizeof(int));
    cudaMalloc(&annd_device, size_of_ann * sizeof(float));
    cudaMalloc(&ann_device, size_of_ann * sizeof(unsigned int));


    ///copy data from host to device
    cudaMemcpy(source_device, source_host,  size_of_source * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(target_device, target_host, size_of_target * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ann_device, ann_host,  size_of_ann * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(annd_device, annd_host,  size_of_ann * sizeof(float), cudaMemcpyHostToDevice);


    ///Runing kernel
    kernelPatchMatch <<<blocks,threads>>> (source_device,target_device,ann_device,annd_device,source.rows,
                                           source.cols,target.rows,target.cols,iters,patch_size,INT_MAX);
    cudaSafeCall ( cudaGetLastError () );
    cudaMemcpy(ann_final_host, ann_device, size_of_ann * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(annd_final_host, annd_device, size_of_ann * sizeof(float), cudaMemcpyDeviceToHost);

    //generate reconstructed, ann, annd images for visualization
    cvMatann.create(source.rows,source.cols, CV_8UC3);
    cvMatannd.create(source.rows,source.cols,CV_32FC1);
    reconstructed_image.create(source.rows,source.cols, CV_8UC3);
    for (int r = 0; r < source.rows; r++) {
        for (int c = 0; c < source.cols; c++) {

            int v = ann_final_host[r*source.cols+c];
            int txbest = INT_TO_X(v);
            int tybest = INT_TO_Y(v);
            cvMatann.at<cv::Vec3b>(r,c)[2] = static_cast<unsigned char>(txbest * 255 / source.cols);
            cvMatann.at<cv::Vec3b>(r,c)[1] = static_cast<unsigned char>(tybest * 255 / source.rows);
            cvMatann.at<cv::Vec3b>(r,c)[0] = 255 - std::max(cvMatann.at<cv::Vec3b>(r,c)[2],
                                            cvMatann.at<cv::Vec3b>(r,c)[1]);
            reconstructed_image.at<cv::Vec3b>(r, c) = target.at<cv::Vec3b>(tybest, txbest);
            cvMatannd.at<float>(r,c) = annd_final_host[r*source.cols+c];
        }
    }


    cudaFree(target_device);
    cudaFree(source_device);
    cudaFree(ann_device);
    cudaFree(annd_device);
}




}

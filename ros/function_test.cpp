#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <curand_kernel.h>
#include <sys/time.h>
#include <chrono>
#include <ros/ros.h>
#include "ros_patch_match/PatchMatchService.h"
#include "ros_patch_match/cuda_functions.cuh"

void convertcvMat2Array(const cv::Mat& img,int*& array)
{
    array = new int[img.rows*img.cols * 3];
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

int main(int argc, char** argv)
{
    cv::Mat mat = cv::imread("Source.jpg",0);
    int* host;
    convertcvMat2Array(mat,host);
    for (int i=0;i<mat.cols;i++)
      for (int j=0;j<mat.rows;j++)  {
          int index = i*mat.rows+j;
        std::cout<<*(host+index)<<std::endl;
    }
    return 0;
}

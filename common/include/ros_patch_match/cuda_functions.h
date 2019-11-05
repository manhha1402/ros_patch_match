#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "ros_patch_match/convenience.cuh"
#include <map>
#include <opencv2/opencv.hpp>

namespace Cuda {

void hostPatchMatch(const cv::Mat& source, const cv::Mat& target,const int iters, const int patch_size,
                    std::string& ann_file);
}

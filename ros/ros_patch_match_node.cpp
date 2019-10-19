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

int max(int x, int y, int z) {
    return std::max(std::max(x, y), z);
}

int min(int x, int y, int z){
    return std::min(std::min(x, y), z);
}
int randomInt(int min, int max) {
    return (int)(((double)rand()/(RAND_MAX+1.0)) * (max - min + 1) + min);
}

class PatchMatch
{
public:
    typedef std::shared_ptr<PatchMatch> Ptr;
    explicit PatchMatch(ros::NodeHandle& node);
    virtual ~PatchMatch();
    bool patchMatchCallback(ros_patch_match::PatchMatchService::Request& req,
                            ros_patch_match::PatchMatchService::Response& res);
    void initAnnGPU(unsigned int *& ann, float *& annd, int aw, int ah,
                    int bw, int bh, int a_cols, int a_rows, int b_cols, int b_rows,
                    const unsigned char * a,const unsigned char * b, int patch_size);
    void initAnnCPU(cv::Mat& ann, cv::Mat& annd,const cv::Mat& source, const cv::Mat& target,
                    const int patch_size);
    cv::Mat ann2image(const cv::Mat& ann);
private:
    ros::NodeHandle node_;
    ros::ServiceServer service_;
    float distance(const cv::Mat& source, const cv::Mat& target,const cv::Mat& mask,
                   const int sx, const int sy,const int tx,const int ty,
                   const int halfpatch_size, const float threshold);



};
PatchMatch::PatchMatch(ros::NodeHandle& node):node_(node)
{
    service_ = node .advertiseService("patch_match_srv",&PatchMatch::patchMatchCallback,this);
}
PatchMatch::~PatchMatch()
{
}
//TODOs
float PatchMatch::distance(const cv::Mat& source, const cv::Mat& target,const cv::Mat& mask,
                           const int sx, const int sy,const int tx,const int ty,
                           const int half_patch_size, const float threshold)
{
    // Do not use patches on boundaries
    if (tx < half_patch_size || tx>= target.cols-half_patch_size ||
            ty < half_patch_size || ty >= target.rows-half_patch_size) {
        return HUGE_VAL;
    }
    // Compute distance between 2 patches S, T
    // Average L2 distance in RGB space
    float pixel_sum = 0, pixel_no = 0, pixel_dist=0;//number of pixels realy counted

    int x1 = max(-half_patch_size,-sx,-tx);
    int x2 = min(half_patch_size, -sx+source.cols-1, -tx+target.cols-1);
    int y1 = max(-half_patch_size, -sy, -ty);
    int y2 = min(half_patch_size, -sy+source.rows-1, -ty+target.rows-1);

    for (int y = y1;y <= y2; y++)
        for (int x = x1; x<= x2; x++)
        {
            cv::Vec3b color_source = source.at<cv::Vec3b>(sy+y,sx+x);
            cv::Vec3b color_target = source.at<cv::Vec3b>(ty+y,tx+x);
            int dr = std::abs(color_source.val[2] - color_target.val[2]);
            int dg = std::abs(color_source.val[1] - color_target.val[1]);
            int db = std::abs(color_source.val[0] - color_target.val[0]);
            pixel_sum =  (float)(dr*dr + dg*dg + db*db);
            pixel_no += 1;
            // Early termination
            //if (pixel_sum > threshold) {return HUGE_VAL;}

        }
    pixel_dist = pixel_sum / pixel_no;
    if (pixel_dist >= threshold) { return threshold; }
    else {
        return pixel_dist;
    }
}

cv::Mat PatchMatch::ann2image(const cv::Mat& ann)
{
    cv::Mat img(ann.rows, ann.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Rect rect(cv::Point(0, 0), ann.size());
    for (int r = 0; r < ann.rows; r++) {
        for (int c = 0; c < ann.cols; c++) {
            img.at<cv::Vec3b>(r,c)[2] = int(ann.at<cv::Vec2i>(r,c)[0] * 255 / ann.cols);
            img.at<cv::Vec3b>(r,c)[1] = int(ann.at<cv::Vec2i>(r,c)[1] * 255 / ann.rows);
            img.at<cv::Vec3b>(r,c)[0] = 255 - std::max(img.at<cv::Vec3b>(r,c)[2],
                    img.at<cv::Vec3b>(r,c)[1]);
        }
    }
    return  img;
}




void PatchMatch::initAnnCPU(cv::Mat& ann, cv::Mat& annd,const cv::Mat& source,
                            const cv::Mat& target, const int patch_size)
{
    ann.create(source.rows,source.cols,CV_32SC2);
    annd.create(source.rows,source.cols,CV_32FC1);
    // randomly initialize the nnf
    srand(time(NULL));
    int half_patch_size = patch_size/2;
#pragma omp parallel for schedule(static)
    for (int sy = 0; sy < source.rows; sy++) {
        for(int sx = 0; sx < source.cols; sx++){

            int tx = randomInt(half_patch_size, target.cols-half_patch_size-1);
            int ty = randomInt(half_patch_size, target.rows-half_patch_size-1);
            ann.at<cv::Vec2i>(sy,sx)[0] = tx;
            ann.at<cv::Vec2i>(sy,sx)[1] = ty;
            annd.at<float>(sy,sx) = distance(source,target,cv::Mat(),sx,sy,tx,ty,half_patch_size,HUGE_VAL);
        }
    }
}





bool PatchMatch::patchMatchCallback(ros_patch_match::PatchMatchService::Request& req,
                                    ros_patch_match::PatchMatchService::Response& res)
{

    cv::Mat source = cv::imread(req.source_file);
    cv::Mat target = cv::imread(req.target_file);
    //judge whether it is empty
    if (source.empty() || target.empty())
    {
        ROS_WARN("PatchMatch::images cannot read!");
        return false;
    }
    ///CPU
    cv::Mat ann,annd;
    int half_patch_size = req.patch_size/2;
    initAnnCPU(ann,annd,source,target,req.patch_size);

    bool forward_search = true;
    for(int i=0;i< req.iters;i++)
    {
        if(forward_search)
        {
#pragma omp parallel for schedule(static)
            // Forward propagation - compare left, center and up
            for (int sy = 1;sy < source.rows;sy++)
            {
                for (int sx = 1;sx < source.cols;sx++)
                {

                    if(annd.at<float>(sy,sx)>0)
                    {
                        //LEFT
                        int tx_left = ann.at<cv::Vec2i>(sy,sx-1)[0] + 1;
                        int ty_left = ann.at<cv::Vec2i>(sy,sx-1)[1];

                        float dist_left = distance(source,target,cv::Mat(),sx,sy,
                                                   tx_left,ty_left,half_patch_size,annd.at<float>(sy,sx));
                        if(dist_left<annd.at<float>(sy,sx))
                        {
                            ann.at<cv::Vec2i>(sy,sx)[0] = tx_left;
                            ann.at<cv::Vec2i>(sy,sx)[1] = ty_left;
                            annd.at<float>(sy,sx) = dist_left;
                        }

                        //UP
                        int tx_up = ann.at<cv::Vec2i>(sy-1,sx)[0];
                        int ty_up = ann.at<cv::Vec2i>(sy-1,sx)[1]+1;

                        float dist_up = distance(source,target,cv::Mat(),sx,sy,
                                                 tx_up,ty_up,half_patch_size,annd.at<float>(sy,sx));
                        if(dist_up<annd.at<float>(sy,sx))
                        {
                            ann.at<cv::Vec2i>(sy,sx)[0] = tx_up;
                            ann.at<cv::Vec2i>(sy,sx)[1] = ty_up;
                            annd.at<float>(sy,sx) = dist_up;
                        }

                    }
                }
            }
        }
        else {
#pragma omp parallel for schedule(static)
            // Backward propagation - compare right, center and down
            for (int sy = source.rows-2 ;sy >=0 ; sy--)
            {
                for (int sx = source.cols - 2;sx >= 0; sx--)
                {
                    if(annd.at<float>(sy,sx)>0)
                    {
                        //RIGHT
                        int tx_right = ann.at<cv::Vec2i>(sy,sx+1)[0] - 1;
                        int ty_right = ann.at<cv::Vec2i>(sy,sx+1)[1];
                        float dist_right = distance(source,target,cv::Mat(),sx,sy,tx_right,ty_right,
                                                    half_patch_size,annd.at<float>(sy,sx));
                        if(dist_right<annd.at<float>(sy,sx))
                        {
                            ann.at<cv::Vec2i>(sy,sx)[0] = tx_right;
                            ann.at<cv::Vec2i>(sy,sx)[1] = ty_right;
                            annd.at<float>(sy,sx) = dist_right;

                        }
                        //DOWN
                        int tx_down = ann.at<cv::Vec2i>(sy+1,sx)[0];
                        int ty_down = ann.at<cv::Vec2i>(sy+1,sx)[1]-1;
                        float dist_down = distance(source,target,cv::Mat(),sx,sy,tx_down,ty_down,
                                                   half_patch_size,annd.at<float>(sy,sx));
                        if(dist_down<annd.at<float>(sy,sx))
                        {
                            ann.at<cv::Vec2i>(sy,sx)[0] = tx_down;
                            ann.at<cv::Vec2i>(sy,sx)[1] = ty_down;
                            annd.at<float>(sy,sx) = dist_down;
                        }
                    }
                }
            }
        }

        forward_search = !forward_search;
#pragma omp parallel for schedule(static)
        //Random search
        for (int sy = 0;sy < source.rows;sy++)
        {
            for (int sx = 0;sx < source.cols;sx++)
            {
                if(annd.at<float>(sy,sx)>0)
                {
                    //get high dimension as radius
                    int radius = target.cols > target.rows ? target.cols : target.rows;

                    // search an exponentially smaller window each iteration
                    while (radius > 8) {
                        // Search around current offset vector (distance-weighted)

                        // clamp the search window to the image
                        int min_tx = ann.at<cv::Vec2i>(sy,sx)[0] - radius;
                        int max_tx = ann.at<cv::Vec2i>(sy,sx)[0] + radius+1;

                        int min_ty = ann.at<cv::Vec2i>(sy,sx)[1] - radius;
                        int max_ty = ann.at<cv::Vec2i>(sy,sx)[1] + radius+1;

                        if (min_tx < 0) { min_tx = 0; }
                        if (max_tx > target.cols) { max_tx = target.cols; }
                        if (min_ty < 0) { min_ty = 0; }
                        if (max_ty > target.rows) { max_ty = target.rows; }

                        int rand_x = randomInt(min_tx,max_tx-1);
                        int rand_y = randomInt(min_ty,max_ty-1);
                        float dist = distance(source,target,cv::Mat(),sx,sy,rand_x,rand_y,
                                              half_patch_size,annd.at<float>(sy,sx));

                        if(dist<annd.at<float>(sy,sx))
                        {
                            ann.at<cv::Vec2i>(sy,sx)[0] = rand_x;
                            ann.at<cv::Vec2i>(sy,sx)[1] = rand_y;
                            annd.at<float>(sy,sx) = dist;
                        }
                        radius >>=1;
                    }

                }
            }
        }

    }

    annd.convertTo(annd,CV_16UC1,1000);
    cv::imwrite("annd.png",annd);
    cv::Mat ann_img = ann2image(ann);
    cv::imwrite("ann.png",ann_img);
    cv::Mat reconstruction(target.rows, target.cols, CV_8UC3);
    for(int i = 0; i < target.rows; i++){
        for(int j = 0; j < target.cols; j++){
            cv::Point p = ann.at<cv::Point>(i,j);

            cv::Vec3b color = source.at<cv::Vec3b>(p);

            reconstruction.at<cv::Vec3b>(i,j) = color;
        }
    }
    cv::imwrite("reconstruction.png",reconstruction);

    /* GPU
    unsigned char *source_device, *target_device;
    int *params_host, *params_device;


    const int patch_w = 3;
    int pm_iters = 5;
    int sizeOfParams = 7;
    int rs_max = INT_MAX;
    int sizeOfAnn;

    //convert cvMat to array
    unsigned char *source_host = source.isContinuous()? source.data: source.clone().data;
    unsigned char *target_host = target.isContinuous()? target.data: target.clone().data;

    //initialization
    unsigned int *ann_host = (unsigned int *)malloc(source.cols*source.rows * sizeof(unsigned int));
    float *annd_host = (float *)malloc(source.cols*source.rows * sizeof(float));
    unsigned int *newann = (unsigned int *)malloc(source.cols*source.rows * sizeof(unsigned int));
    float *newannd = (float *)malloc(source.cols*source.rows * sizeof(float));
    */
    return  true;
}

void PatchMatch::initAnnGPU(unsigned int *& ann, float *& annd, int aw, int ah,
                            int bw, int bh, int a_cols, int a_rows, int b_cols, int b_rows,
                            const unsigned char * a,const unsigned char * b,int patch_size)
{
    for (int ay = 0; ay < a_rows; ay++) {
        for (int ax = 0; ax < a_cols; ax++) {
            int bx = rand() % b_cols;
            int by = rand() % b_rows;

            ann[ay*a_cols + ax] = XY_TO_INT(bx, by);
            //annd[ay*a_cols + ax] = Cuda::dist(a, b,);

        }
    }
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "ros_patch_match_node");
    ros::NodeHandle node;
    PatchMatch::Ptr patch_match = PatchMatch::Ptr(new PatchMatch(node));
    ros::spin();
    return 0;
}

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
    ///
    /// \brief reconstructImageCPU reconstruct source image from patches of target image
    /// \param source original source image
    /// \param target target image
    /// \param ann approximate search
    /// \return new source image
    ///
    cv::Mat reconstructImageCPU(const cv::Mat& source, const cv::Mat& target,const cv::Mat& ann);
    cv::Mat reconstructError(const cv::Mat& source,const cv::Mat target,const cv::Mat& ann);
private:
    ros::NodeHandle node_;
    ros::ServiceServer service_;
    float distance(const cv::Mat& source, const cv::Mat& target,
                   const int sx, const int sy,const int tx,const int ty,
                   const int half_patch_size, const float threshold);
    void improveGuess(const cv::Mat& source,const cv::Mat& target,
                      const int sx,const int sy,const int tx,const int ty,const int half_patch_size,
                      int& tx_best, int& ty_best,float& dist_best);


};
PatchMatch::PatchMatch(ros::NodeHandle& node):node_(node)
{
    service_ = node .advertiseService("patch_match_srv",&PatchMatch::patchMatchCallback,this);
}
PatchMatch::~PatchMatch()
{
}
//TODOs
float PatchMatch::distance(const cv::Mat& source, const cv::Mat& target,
                           const int sx, const int sy,const int tx,const int ty,
                           const int half_patch_size, const float threshold)
{

    // Compute distance between 2 patches S, T
    // Average L2 distance in RGB space
    float ans = 0, num = 0;


    for (int y = -half_patch_size;y <= half_patch_size; y++)
        for (int x = -half_patch_size; x<= half_patch_size; x++)
        {
            if (
                    (sy + y) < source.rows && (sy + y) >= 0 && (sx + x) < source.cols && (sx + x) >= 0
                    &&
                    (ty + y) < target.rows && (ty + y) >= 0 && (tx + x) < target.cols && (tx + x) >= 0
                    )//the pixel in source and target should exist
            {
                cv::Vec3b color_source = source.at<cv::Vec3b>(sy+y,sx+x);
                cv::Vec3b color_target = target.at<cv::Vec3b>(ty+y,tx+x);
                int dr = std::abs(color_source.val[2] - color_target.val[2]);
                int dg = std::abs(color_source.val[1] - color_target.val[1]);
                int db = std::abs(color_source.val[0] - color_target.val[0]);
                ans +=  (float)(dr*dr + dg*dg + db*db);
                num += 1;
            }

        }
    ans = ans / num;
    if (ans >= threshold) { return threshold; }
    else {
        return ans;
    }
}
void PatchMatch::improveGuess(const cv::Mat& source,const cv::Mat& target,
                              const int sx,const int sy,const int tx,const int ty,const int half_patch_size,
                              int& tx_best, int& ty_best,float& dist_best)
{
    float d = 0;
    d = distance(source,target,sx,sy,tx,ty,half_patch_size,dist_best);
    if (d < dist_best) {
        tx_best = tx;
        ty_best = ty;
        dist_best = d;
    }
}


cv::Mat PatchMatch::ann2image(const cv::Mat& ann)
{
    cv::Mat img(ann.rows, ann.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Rect rect(cv::Point(0, 0), ann.size());

#pragma omp parallel for schedule(static)
    for (int r = 0; r < ann.rows; r++) {
        for (int c = 0; c < ann.cols; c++) {

            int v = ann.at<int>(r,c);
            int xbest = INT_TO_X(v);
            int ybest = INT_TO_Y(v);
            img.at<cv::Vec3b>(r,c)[2] = int(xbest * 255 / ann.cols);
            img.at<cv::Vec3b>(r,c)[1] = int(ybest * 255 / ann.rows);
            img.at<cv::Vec3b>(r,c)[0] = 255 - std::max(img.at<cv::Vec3b>(r,c)[2],img.at<cv::Vec3b>(r,c)[1]);
        }
    }
    return  img;
}




void PatchMatch::initAnnCPU(cv::Mat& ann, cv::Mat& annd,const cv::Mat& source,
                            const cv::Mat& target, const int patch_size)
{
    ann.create(source.rows,source.cols,CV_32SC1);
    annd.create(source.rows,source.cols,CV_32FC1);
    // randomly initialize the nnf
    srand(time(NULL));
    int half_patch_size = patch_size/2;
#pragma omp parallel for schedule(static)
    for (int sy = 0; sy < source.rows; sy++) {
        for(int sx = 0; sx < source.cols; sx++){

            int tx =  rand() % target.cols;
            int ty =  rand() % target.rows;
            ann.at<int>(sy,sx) = XY_TO_INT(tx,ty);
            annd.at<float>(sy,sx) = distance(source,target,sx,sy,tx,ty,half_patch_size,INT_MAX);
        }
    }
}
void PatchMatch::initAnnGPU(unsigned int *& ann, float *& annd, int aw, int ah,
                            int bw, int bh, int a_cols, int a_rows, int b_cols, int b_rows,
                            const unsigned char * a,const unsigned char * b,int patch_size)
{

#pragma omp parallel for schedule(static)
    for (int ay = 0; ay < a_rows; ay++) {
        for (int ax = 0; ax < a_cols; ax++) {
            int bx = rand() % b_cols;
            int by = rand() % b_rows;

            ann[ay*a_cols + ax] = XY_TO_INT(bx, by);
            //annd[ay*a_cols + ax] = Cuda::dist(a, b,);

        }
    }
}

cv::Mat PatchMatch::reconstructImageCPU(const cv::Mat& source, const cv::Mat& target,const cv::Mat& ann)
{
    cv::Mat source_recon;
    source.copyTo(source_recon);
#pragma omp parallel for schedule(static)
    for (int sy = 0; sy < source.rows; sy++) {
        for (int sx = 0; sx < source.cols; sx++)
        {

            int p = ann.at<int>(sy,sx);
            int xbest = INT_TO_X(p);
            int ybest = INT_TO_Y(p);
            cv::Vec3b bi = target.at<cv::Vec3b>(ybest, xbest);
            source_recon.at<cv::Vec3b>(sy, sx).val[2] = bi.val[2];
            source_recon.at<cv::Vec3b>(sy, sx).val[1] = bi.val[1];
            source_recon.at<cv::Vec3b>(sy, sx).val[0] = bi.val[0];
        }
    }
    return source_recon;
}

cv::Mat PatchMatch::reconstructError(const cv::Mat& source,const cv::Mat target,const cv::Mat& ann) {
    cv::Mat c;
    source.copyTo(c);
    int * err, err_min=INT_MAX, err_max=0;
    err = new int[source.rows*source.cols];
#pragma omp parallel for schedule(static)
    for (int sy = 0; sy < source.rows; sy++) {
        for (int sx = 0; sx < source.cols; sx++)
        {
            int v = ann.at<int>(sy,sx);
            int xbest = INT_TO_X(v);
            int ybest = INT_TO_Y(v);
            cv::Vec3b bi = target.at<cv::Vec3b>(ybest, xbest);
            cv::Vec3b ai = source.at<cv::Vec3b>(sy, sx);
            int err_0 = ai.val[0] - bi.val[0];
            int err_1 = ai.val[1] - bi.val[1];
            int err_2 = ai.val[2] - bi.val[2];
            int error = err_0*err_0 + err_1*err_1 + err_2*err_2;
            if (error<err_min)
            {
                err_min = error;
            }
            if (error>err_max)
            {
                err_max = error;
            }

        }
    }
#pragma omp parallel for schedule(static)
    for (int sy = 0; sy < source.rows; sy++) {
        for (int sx = 0; sx < source.cols; sx++)
        {
            c.at<cv::Vec3b>(sy, sx).val[1] = (uchar)255 * ((float)(err[sy*source.cols+sx]-err_min)/(err_max-err_min));
            c.at<cv::Vec3b>(sy, sx).val[2] = 0;
            c.at<cv::Vec3b>(sy, sx).val[0] = 0;
        }
    }

    return c;
}


bool PatchMatch::patchMatchCallback(ros_patch_match::PatchMatchService::Request& req,
                                    ros_patch_match::PatchMatchService::Response& res)
{

    cv::Mat source = cv::imread(req.source_file);
    cv::Mat target = cv::imread(req.target_file);
    //judge whether it is empty
    if (source.empty() || target.empty())
    {
        ROS_WARN("PatchMatch::patchMatchCallback image cannot be read!");
        return false;
    }
    if(req.patch_size % 2 == 0)
    {
        ROS_WARN("PatchMatch::patchMatchCallback patch size must be odd !");
        return false;
    }
    ///CPU
    cv::Mat ann,annd;
    int half_patch_size = req.patch_size/2; //should be odd
    if(bool(req.use_gpu)==0)
    {
        ROS_INFO("PatchMatch CPU Version");
        initAnnCPU(ann,annd,source,target,req.patch_size);
        for(int iter=0;iter< req.iters;iter++)
        {
            //Forward search: UP and LEFT
            /* In each iteration, improve the NNF, by looping in scanline or reverse-scanline order. */
            //UP AND LEFT for even iterations
            int ystart = 0, yend = source.rows, ychange = 1;
            int xstart = 0, xend = source.cols, xchange = 1;

            //BELOW AND RIGHT for odd iterations
            if (iter % 2 == 1) {
                xstart = xend-1; xend = -1; xchange = -1;
                ystart = yend-1; yend = -1; ychange = -1;
            }
#pragma omp parallel for schedule(static)
            for (int sy = ystart; sy != yend; sy += ychange) {
                for (int sx = xstart; sx != xend; sx += xchange) {
                    /* Current (best) guess. */
                    int v = ann.at<int>(sy,sx);
                    int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
                    float dbest = annd.at<float>(sy,sx);
                    // Propagation: Improve current guess by trying instead correspondences from left and up (even iteration)
                    if ((unsigned)(sx - xchange) < (unsigned)source.cols && (sx - xchange) >= 0) {

                        int vp = ann.at<int>(sy,sx - xchange);
                        int tx = INT_TO_X(vp) + xchange, ty = INT_TO_Y(vp);
                        if ((unsigned)tx < (unsigned)target.cols) {
                            improveGuess(source, target,sx,sy,tx,ty,half_patch_size,xbest,ybest,dbest);
                        }
                    }
                    //Propagation: below and right on odd iterations.
                    if ((unsigned)(sy - ychange) < (unsigned)source.rows && (sy - ychange) >= 0) {
                        int vp = ann.at<int>(sy-ychange,sx);
                        int tx = INT_TO_X(vp), ty = INT_TO_Y(vp) + ychange;
                        if ((unsigned)ty < (unsigned)target.rows) {
                            improveGuess(source, target,sx,sy,tx,ty,half_patch_size,xbest,ybest,dbest);
                        }
                    }
                    /* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
                    int rs_start = INT_MAX;
                    if (rs_start > MAX(target.cols, target.rows))
                    { rs_start = MAX(target.cols, target.rows); }
                    for (int mag = rs_start; mag >= 1; mag /= 2) {
                        /* Sampling window */
                        int xmin = MAX(xbest - mag, 0), xmax = MIN(xbest + mag + 1, target.cols);
                        int ymin = MAX(ybest - mag, 0), ymax = MIN(ybest + mag + 1, target.rows);
                        int tx = xmin + rand() % (xmax - xmin);
                        int ty = ymin + rand() % (ymax - ymin);
                        improveGuess(source, target,sx,sy,tx,ty,half_patch_size,xbest,ybest,dbest);
                    }
                    ann.at<int>(sy,sx) = XY_TO_INT(xbest, ybest);
                    annd.at<float>(sy,sx) = dbest;
                }
            }
        }
        cv::Mat reconstructed_image = reconstructImageCPU(source,target,ann);
        cv::Mat error_map = reconstructError(source,target,ann);
        cv::Mat ann_map = ann2image(ann);
        cv::imwrite("reconstructed_image.png",reconstructed_image);
        cv::imwrite("error_map.png",error_map);
        cv::imwrite("ann_map.png",ann_map);
    }
    else
    {
        ROS_INFO("PatchMatch GPU Version");
        unsigned char *source_device, *target_device;
        int *params_host, *params_device;
        const int patch_w = 3;
        int pm_iters = 5;
        int sizeOfParams = 7;


        //convert cvMat to array
        unsigned char *source_host = source.isContinuous()? source.data: source.clone().data;
        unsigned char *target_host = target.isContinuous()? target.data: target.clone().data;

        //initialization
        unsigned int *ann_host = (unsigned int *)malloc(source.cols*source.rows * sizeof(unsigned int));
        float *annd_host = (float *)malloc(source.cols*source.rows * sizeof(float));
        unsigned int *new_ann = (unsigned int *)malloc(source.cols*source.rows * sizeof(unsigned int));
        float *new_annd = (float *)malloc(source.cols*source.rows * sizeof(float));

    }

    ROS_INFO("PatchMatch succeeded !");
    return  true;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "ros_patch_match_node");
    ros::NodeHandle node;
    PatchMatch::Ptr patch_match = PatchMatch::Ptr(new PatchMatch(node));
    ros::spin();
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/filesystem.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include "ros_patch_match/PatchMatchService.h"
#include "ros_patch_match/convenience.cuh"
int randomInt(int min, int max) {
    return (int)(((double)rand()/(RAND_MAX+1.0)) * (max - min + 1) + min);
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "ros_patch_match_client_node");
    ros::NodeHandle node;
    ros::ServiceClient client = node.serviceClient<ros_patch_match::PatchMatchService>("patch_match_srv");

    ros_patch_match::PatchMatchService srv;
    srv.request.iters = 5;
    srv.request.use_gpu = true;
    srv.request.patch_size = 7;


    std::vector<cv::String> source_files,target_files;
    std::string data_path = ros::package::getPath("ros_patch_match")+"/color_map_data/";


    // Get all jpg in the folder
    cv::glob(data_path+"*.jpg",source_files);

    srv.request.source_file = source_files[0];
    srv.request.target_file = source_files[1];
    srv.request.ann_file = data_path+"ann.txt";
    //srv.request.reconstructed_image_file = data_path+ss.str();
    client.waitForExistence();
    client.call(srv);
    /*
    target_files = source_files;

    for (int i=0;i<source_files.size();i++) {
        for (int j=0;j<target_files.size();j++) {
            std::cout<<"process "<<i<<":"<<j<<std::endl;
            std::ostringstream ss;
            ss<<i<<"_"<<j;
            srv.request.source_file = source_files[i];
            srv.request.target_file = target_files[j];
            srv.request.ann_file = data_path+ss.str();
            srv.request.annd_file = data_path+ss.str();
            srv.request.reconstructed_image_file = data_path+ss.str();
            client.waitForExistence();
            client.call(srv);
        }
    }
    */
    cv::Mat source = cv::imread(source_files[0]);
    cv::Mat target = cv::imread(source_files[1]);
    cv::Mat ann(source.rows,source.cols,CV_8UC3),reconstructed_image(source.rows,source.cols,CV_8UC3);
    std::ifstream infile(srv.request.ann_file.c_str(), std::ios::in | std::ios::binary);
    if ( !infile.is_open() ) {
        std::cout<<"!!! Open file [" + srv.request.ann_file + "] failed!"<<std::endl;
        return 0;
    }
    for( int r = 0; r < source.rows; r++ ){
        for( int c = 0; c < source.cols; c++ ){
            int txbest,tybest;
            float annd_value;
            infile>>txbest>>tybest>>annd_value;
            ann.at<cv::Vec3b>(r,c)[2] = static_cast<unsigned char>(txbest * 255 / source.cols);
            ann.at<cv::Vec3b>(r,c)[1] = static_cast<unsigned char>(tybest * 255 / source.rows);
            ann.at<cv::Vec3b>(r,c)[0] = 255 - std::max(ann.at<cv::Vec3b>(r,c)[2],
                    ann.at<cv::Vec3b>(r,c)[1]);
            reconstructed_image.at<cv::Vec3b>(r, c) = target.at<cv::Vec3b>(tybest, txbest);
        }
    }
    infile.close();
    cv::imwrite("recon.png",reconstructed_image);
    cv::imwrite("ann.png",ann);
    return 0;
}

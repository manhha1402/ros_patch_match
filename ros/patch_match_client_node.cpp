#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/filesystem.hpp>
#include <ros/ros.h>
#include <ros/package.h>
#include "ros_patch_match/PatchMatchService.h"

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

    return 0;
}

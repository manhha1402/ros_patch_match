cmake_minimum_required(VERSION 2.8.3)
project(ros_patch_match)
add_compile_options(-std=c++14)

set(catkin_RUN_PACKAGES
    message_generation
    roscpp
    roslaunch
    roslib
    std_msgs)
# set catkin packages that are needed when building the code
set(catkin_BUILD_PACKAGES
    ${catkin_RUN_PACKAGES}
    cmake_modules
    )
## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
    ${catkin_BUILD_PACKAGES}
    )
find_package(CUDA REQUIRED)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++14 -D_MWAITXINTRIN_H_INCLUDED -lcudart -lcublas -lcurand -gencode arch=compute_61,code=sm_61)

find_package(catkin REQUIRED COMPONENTS ${catkin_RUN_PACKAGES})
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(OpenCV 3 REQUIRED)
add_service_files(
   FILES
   PatchMatchService.srv
)
generate_messages(

    DEPENDENCIES std_msgs
)
catkin_package(
    INCLUDE_DIRS common/include
    CATKIN_DEPENDS roscpp roslaunch roslib message_runtime std_msgs 
    DEPENDS
    LIBRARIES
)
include_directories(
    common/include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node ros/ros_patch_match_node.cpp)
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

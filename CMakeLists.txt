cmake_minimum_required(VERSION 2.8.3)
project(ros_patch_match)
add_compile_options(-std=c++14)
find_package(OpenMP)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
SET(CMAKE_BUILD_TYPE Release)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(catkin_RUN_PACKAGES
    message_generation
    message_runtime
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
include("${CMAKE_SOURCE_DIR}/cmake/find_cuda.cmake")


find_package(catkin REQUIRED COMPONENTS ${catkin_RUN_PACKAGES})
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(OpenCV REQUIRED NO_MODULE PATHS /usr/local NO_DEFAULT_PATH)

add_service_files(
    FILES
    PatchMatchService.srv
    )
generate_messages(
    DEPENDENCIES std_msgs
    )
catkin_package(
    INCLUDE_DIRS common/include
    CATKIN_DEPENDS ${catkin_BUILD_PACKAGES}
    DEPENDS
    LIBRARIES ${PROJECT_NAME}
    )
include_directories(
    common/include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    )

# make sure to copy the license lib prior to building the verifier
file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_INCLUDE_DESTINATION})
file(MAKE_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_LIB_DESTINATION})


file(GLOB CUDA_SRCS "${PROJECT_SOURCE_DIR}/common/src/cuda/*.cu")
cuda_add_library(${PROJECT_NAME} "${CUDA_SRCS}")
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


add_executable(${PROJECT_NAME}_node ros/ros_patch_match_node.cpp)
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME}  ${catkin_LIBRARIES})
### Client ####
add_executable(patch_match_client_node ros/patch_match_client_node.cpp)
add_dependencies(patch_match_client_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(patch_match_client_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

#############
## Install ##
#############
## Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}_node patch_match_client_node
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
install(DIRECTORY common/include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.cuh"
    PATTERN ".svn" EXCLUDE
    )




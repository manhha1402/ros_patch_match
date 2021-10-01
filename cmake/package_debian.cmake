# general cpack variables
set(CPACK_PACKAGE_CONTACT "Manh Ha Hoang <manhha.hoang@neura-robotics.com>")
set(CPACK_PACKAGE_VENDOR "Neura Robotics GmbH, Metzingen, Germany")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Neura Robotics ${PROJECT_NAME} package")

# CPACK_PACKAGE_VERSION
if (PACKAGE_VERSION)
    set(CPACK_PACKAGE_VERSION ${PACKAGE_VERSION})
else ()
    message(WARNING "PACKAGE_VERSION not set! Did you include project_version.cmake?")
    if (PROJECT_VERSION)
        message(WARNING "CPACK_PACKAGE_VERSION: Falling back to PROJECT_VERSION (${PROJECT_VERSION})")
        set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
    elseif (PROJECT_VERSION)
        message(WARNING "CPACK_PACKAGE_VERSION: Falling back to PROJECT_VERSION (${PROJECT_VERSION})")
        set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
    endif ()
endif ()

###############################
# debian package specific stuff
###############################
set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_DEBUG ON)
set(CPACK_DEBIAN_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION})
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
#Debian dependency packages
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libyaml-cpp-dev, ros-$ENV{ROS_DISTRO}-roscpp, ros-$ENV{ROS_DISTRO}-roslib")

# package name defaults to lower case of project name with _ replaced by -
if (NOT CPACK_PACKAGE_NAME)
    string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWER)
    string(REPLACE "_" "-" CPACK_PACKAGE_NAME "${PROJECT_NAME_LOWER}")
endif ()
message(WARNING "CPACK_PACKAGE_NAME: " ${CPACK_PACKAGE_NAME})

# check if it is a ROS/catkin package
if (EXISTS "${PROJECT_SOURCE_DIR}/package.xml")
    set(ROS_DISTRO $ENV{ROS_DISTRO})
    if (ROS_DISTRO)
        set(CPACK_PACKAGE_NAME "ros-${ROS_DISTRO}-${CPACK_PACKAGE_NAME}")
        set(CPACK_SET_DESTDIR true)
    else ()
        message(WARNING "ROS_DISTRO not set. Not treating this as a ROS package.")
    endif ()
endif ()

include(CPack)

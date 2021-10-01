#
# Sets PROJECT_VERSION and PACKAGE_VERSION
#

# Split a version number into separate components
# version the version number to split
# major variable name to store the major version in
# minor variable name to store the minor version in
# patch variable name to store the patch version in
# extra variable name to store a version suffix in
function(version_split version major minor patch extra)
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*)?" version_valid ${version})
    if(version_valid)
        string(REGEX    REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)(.*)?" "\\1;\\2;\\3;\\4" VERSION_MATCHES ${version})
        list(GET VERSION_MATCHES 0 version_major)
        set(${major} ${version_major} PARENT_SCOPE)
        list(GET VERSION_MATCHES 1 version_minor)
        set(${minor} ${version_minor} PARENT_SCOPE)
        list(GET VERSION_MATCHES 2 version_patch)
        set(${patch} ${version_patch} PARENT_SCOPE)
        list(GET VERSION_MATCHES 3 version_extra)
        set(${extra} ${version_extra} PARENT_SCOPE)
    else(version_valid)
        message(AUTHOR_WARNING "Bad version ${version}; falling back to 0 (have you made an initial release?)")
        set(${major} "0" PARENT_SCOPE)
        set(${minor} "0" PARENT_SCOPE)
        set(${patch} "0" PARENT_SCOPE)
        set(${extra} "" PARENT_SCOPE)
    endif(version_valid)
endfunction(version_split)

#########################
# get PACKAGE_XML_VERSION
#########################
if (EXISTS "${PROJECT_SOURCE_DIR}/package.xml")
    file(STRINGS "${PROJECT_SOURCE_DIR}/package.xml" PACKAGE_XML_VERSION_LINE REGEX <version>[0-9.]*</version>)
    string(REGEX REPLACE .*<version>\([0-9.]*\)</version>.* \\1 PACKAGE_XML_VERSION "${PACKAGE_XML_VERSION_LINE}")
    MESSAGE(WARNING "PACKAGE_XML_VERSION: " ${PACKAGE_XML_VERSION})

endif ()

########################
# set PROJECT_VERSION:
#   - package.xml
#   - default 0.0.0
########################

if (PACKAGE_XML_VERSION)
    set(PROJECT_VERSION ${PACKAGE_XML_VERSION})
    set(PACKAGE_VERSION ${PACKAGE_XML_VERSION})
else ()
    message(WARNING "PACKAGE_VERSION not set. Defaulting to 0.0.0")
    set(PROJECT_VERSION "0.0.0")
endif ()

version_split(${PROJECT_VERSION} PACKAGE_VERSION_MAJOR PACKAGE_VERSION_MINOR PACKAGE_VERSION_PATCH extra)

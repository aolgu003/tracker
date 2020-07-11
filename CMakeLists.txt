project(tracker)

cmake_minimum_required(VERSION 3.5)

find_package(OpenCV 3 REQUIRED)

find_package(GTSAM REQUIRED) # Uses installed package
include_directories(${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBRARIES gtsam)   # TODO: automatic search libs

set(CMAKE_CXX_STANDARD 14)

include(cmake/DownloadProject.cmake)

#---------------- DOWNLOAD EXTERNAL DEPENDENCIES ---------------------------------------
download_project(PROJ                googletest
                 GIT_REPOSITORY      https://github.com/google/googletest.git
                 GIT_TAG             master
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
)

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

# When using CMake 2.8.11 or later, header path dependencies
# are automatically added to the gtest and gmock targets.
# For earlier CMake versions, we have to explicitly add the
# required directories to the header search path ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
                        "${gmock_SOURCE_DIR}/include")
endif()
include_directories(${GTEST_INCLUDE_DIRS})

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

include_directories(.)

add_library( klttracker klttracker.cpp photometric_error_factor.h camera.h)
target_link_libraries( klttracker ${OpenCV_LIBS} )

add_executable( trackerExample trackerExample.cpp )
target_link_libraries( trackerExample klttracker ${OpenCV_LIBS} )

add_executable(frame_alignment_test test/test_frame_alignment.cpp)
target_link_libraries(frame_alignment_test gtest_main gtsam ${OpenCV_LIBS})

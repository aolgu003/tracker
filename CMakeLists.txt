project(tracker)

cmake_minimum_required(VERSION 3.5)

find_package(OpenCV 3 REQUIRED)


add_library( klttracker klttracker.cpp )
target_link_libraries( klttracker ${OpenCV_LIBS} )

add_executable( trackerExample trackerExample.cpp )
target_link_libraries( trackerExample klttracker ${OpenCV_LIBS} )

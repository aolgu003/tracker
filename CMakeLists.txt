add_library( klttracker klttracker.cpp )
target_link_libraries( klttracker ${OpenCV_LIBS} )

add_executable( trackerExample trackerExample.cpp )
target_link_libraries( trackerExample klttracker ${OpenCV_LIBS} )

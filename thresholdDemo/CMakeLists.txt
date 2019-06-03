cmake_minimum_required(VERSION 2.8)
project( thresholdDemo )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( thresholdDemo main.cpp )
target_link_libraries( thresholdDemo ${OpenCV_LIBS} )


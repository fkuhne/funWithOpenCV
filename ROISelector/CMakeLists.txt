cmake_minimum_required(VERSION 2.8)
project( ROISelector )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} .)
add_executable( ROISelector main.cpp ROISelector.cpp )
target_link_libraries( ROISelector ${OpenCV_LIBS} )


cmake_minimum_required(VERSION 2.8)
project( findContours )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( findContours main.cpp )
target_link_libraries( findContours ${OpenCV_LIBS} )


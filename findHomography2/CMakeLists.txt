cmake_minimum_required(VERSION 2.8)
project( findHomography )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( findHomography main.cpp )
target_link_libraries( findHomography ${OpenCV_LIBS} )


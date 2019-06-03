cmake_minimum_required(VERSION 2.8)
project( SURFNaked )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( SURFNaked main.cpp )
target_link_libraries( SURFNaked ${OpenCV_LIBS} )


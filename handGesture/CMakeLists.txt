cmake_minimum_required(VERSION 2.8)
project( handGesture )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( handGesture main.cpp )
target_link_libraries( handGesture ${OpenCV_LIBS} )


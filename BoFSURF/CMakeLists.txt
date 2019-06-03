cmake_minimum_required(VERSION 2.8)
project( BoFSURF )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( BoFSURF main.cpp )
target_link_libraries( BoFSURF ${OpenCV_LIBS} )


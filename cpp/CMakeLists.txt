cmake_minimum_required(VERSION 3.10)

project( plane_sweep )

find_package( OpenCV REQUIRED )

add_executable( plane_sweep plane_sweep.cpp )

target_link_libraries( plane_sweep ${OpenCV_LIBS} )

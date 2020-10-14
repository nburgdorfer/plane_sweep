cmake_minimum_required(VERSION 3.10)

project( plane_sweep )

find_package( OpenCV REQUIRED )

find_package(OpenMP)

add_executable( plane_sweep plane_sweep.cpp util.cpp )

target_link_libraries(plane_sweep PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS} )

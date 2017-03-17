cmake_minimum_required(VERSION 3.6)
project(ocv_glitch_art)
set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES main.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(ocv_glitch_art ${SOURCE_FILES})
target_link_libraries( ocv_glitch_art ${OpenCV_LIBS} )
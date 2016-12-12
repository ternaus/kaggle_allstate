cmake_minimum_required(VERSION 2.8)
project(Iso)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -fexceptions")
set(SOURCE_FILES main.cpp)
add_executable(Iso ${SOURCE_FILES})

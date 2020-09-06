cmake_minimum_required(VERSION 3.10)
project(cpp_mlp)

if(NOT "${CMAKE_CXX_STANDARD}")
  set(CMAKE_CXX_STANDARD 17)
endif()

add_subdirectory(src)

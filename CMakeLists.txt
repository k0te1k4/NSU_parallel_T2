cmake_minimum_required(VERSION 3.26)
project(NSU_parallel_T2 C)

set(CMAKE_C_STANDARD 11)

add_executable(NSU_parallel_T2
        dgemv.c
        integ.c
        lin.cpp)

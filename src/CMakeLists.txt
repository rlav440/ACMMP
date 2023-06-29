cmake_minimum_required (VERSION 2.8)
project (ACMMP_recon)


set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA 6.0 REQUIRED ) # For Cuda Managed Memory and c++11
find_package(OpenCV REQUIRED )
find_package(Boost COMPONENTS program_options REQUIRED)


if (NOT(CMAKE_BUILD_TYPE))
    SET(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
include_directories(.)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_86,code=sm_86)
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

cuda_add_library(
        ${PROJECT_NAME} SHARED
        ACMMP.h
        ACMMP.cpp
        ACMMP.cu
        acmmp_definitions.h
        acmmp_definitions.cpp
)

# For compilation ...
# Specify target & source files to compile it from

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS} ${Boost_LIBRARIES}
    )

cuda_add_executable(
        ACMMP
        main_ACMMP.cpp
        )

target_link_libraries(
        ACMMP ${PROJECT_NAME}
)

add_executable(
        fuse_data
        main_fusion.cpp
)
target_link_libraries(fuse_data ${PROJECT_NAME})
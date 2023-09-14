cmake_minimum_required(VERSION 3.26)
project(My_Project CUDA CXX)
# project(My_Project CUDA CXX)
set(CMAKE_PREFIX_PATH /root/miniconda3/envs/nlp/lib/python3.10/site-packages/torch/share/cmake)
# set(CMAKE_PREFIX_PATH /home/lashi/miniconda3/envs/nlp/lib/python3.11/site-packages/torch/share/cmake)

# find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

macro(add_executable_for_torch target_name)
    add_executable(${target_name} "./csrc/${target_name}.cu")
    set_target_properties(${target_name} PROPERTIES CXX_STANDARD 17 CUDA_ARCHITECTURES 89)
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
    target_link_libraries(${target_name} "${TORCH_LIBRARIES}")
endmacro()


# common to all 
include_directories(${CMAKE_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# transpose.cu
# message("FLAGS  "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
# message("Torch libraries " ${TORCH_LIBRARIES})
# message("Cuda lib" ${CUDA_CUDA_LIB})
# add_executable(transpose ./csrc/transpose.cu)
# set_target_properties(transpose PROPERTIES CXX_STANDARD 17 CUDA_ARCHITECTURES 89)
# target_compile_options(transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
# target_link_libraries(transpose "${TORCH_LIBRARIES}")
add_executable_for_torch(transpose)

# smemShared.cu
add_executable_for_torch(smemSquare)

# smemRect.cu
add_executable_for_torch(smemRect)

#smemReduceInteger.cu
add_executable_for_torch(smemReduceInteger)

#smemTranspose.cu
add_executable_for_torch(smemTranspose)

#index.cu
add_executable_for_torch(index)




# option(SOURCE_FILE_PATH "Specify the source file path")
# option(ENABLE_TORCH "ENABLE Torch Library" OFF)
# # message("Hello " ${CMAKE_PREFIX_PATH})


# find_package(CUDAToolkit REQUIRED)
# if(ENABLE_TORCH)
#     set(CMAKE_PREFIX_PATH /usr/local/lib/python3.10/dist-packages/torch/share/cmake)
#     find_package(Torch REQUIRED)
#     message(STATUS "Torch found")
# endif()


# if(SOURCE_FILE_PATH)
#     cmake_path(GET SOURCE_FILE_PATH FILENAME filename)
#     cmake_path(REMOVE_EXTENSION filename)
#     project(${filename})
#     message("Setting the project name as : " ${filename})
# endif()

# 
# find_package(CUDAToolkit REQUIRED)
# find_package(Torch REQUIRED)

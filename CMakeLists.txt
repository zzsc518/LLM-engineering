# 泛化的
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(oneLLM LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})


list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
find_package(CUDA REQUIRED)

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")	
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
#                      -rdc=true") # not sure the effect of this option, retain it temply

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
message("-- Assign GPU architecture (sm=70 75 80 86)")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

message(STATUS "CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_STANDARD STREQUAL "11")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 头文件
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)

# lib
set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

# 添加头文件目录
include_directories(
  ${COMMON_HEADER_DIRS}
)

# 添加库目录
link_directories(
  ${COMMON_LIB_DIRS}
)

option (PERF
  "measure model inference performance"
  OFF
)
option (PRINT_DATA
  "print kernel output to debug"
  OFF
)
option (SAVE_DATA
  "save kernel output to debug"
  OFF
)
if (PERF)
    add_compile_options(-DPERF)
endif()
if (PRINT_DATA)
    add_compile_options(-DPRINT_DATA)
endif()
if (SAVE_DATA)
    add_compile_options(-DSAVE_DATA)
endif()
#cmake .. -DPRINT_DATA=ON && make
#cmake .. -DPRINT_DATA=ON -DSAVE_DATA=ON && make
#cmake .. -DPERF=ON && make
#cmake .. && make

# 全局搜索并赋给LLM_CXX_SOURCES
file(GLOB_RECURSE LLM_CXX_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE LLM_CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cu)

# OBJECT表示库不会被链接成.a或者.so，而是会编译生成变成.o
# 只会生成一组目标文件.o，这些目标文件是源代码经过编译但未链接的中间产物
# 注意是一组而不是一个.o
# 除了OBJECT，还有STATIC和SHARED
add_library(llmengine OBJECT
           ${LLM_CXX_SOURCES}
           ${LLM_CUDA_SOURCES}
           )

# 编译子目录
# 会递归地查找并处理这两个子目录下的 CMakeLists.txt 文件。
add_subdirectory(src)
add_subdirectory(tests)
# add_subdirectory(examples)

# 构建一个叫main的可执行文件，user_entry.cpp是main函数的入口
add_executable(main user_entry.cpp)

# 给可执行文件链接库
target_link_libraries(main PUBLIC -lcublas -lcudart -lcudadevrt llmengine)

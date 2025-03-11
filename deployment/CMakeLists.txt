cmake_minimum_required(VERSION 3.10)
project(EdgeVoice VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 如果未定义构建类型，设置为Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 设置编译器标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# 设置include目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 设置源文件
set(SOURCES 
    src/audio_processor.cpp
    src/inference_engine.cpp
    src/main.cpp
)

# 创建可执行文件
add_executable(edgevoice ${SOURCES})

# 如果在实际部署中需要链接HiAI库，添加相应的链接
# target_link_libraries(edgevoice ${HIAI_LIBRARIES})

# 设置安装路径
if(NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")
endif()

# 设置安装规则
install(TARGETS edgevoice DESTINATION bin)

# 输出构建信息
message(STATUS "")
message(STATUS "EdgeVoice Configuration Summary")
message(STATUS "------------------------------")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "") 
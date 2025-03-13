# EdgeVoice HiAI部署 CMakeLists.txt
cmake_minimum_required(VERSION 3.8)
project(EdgeVoice VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now -Wl,-z,noexecstack -fPIC -fPIE -pie")

# 添加安全编译选项
add_compile_options(-fstack-protector-all -fPIC)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # HiAI相关头文件目录
    ${HIAI_INCLUDE_DIR}
)

# 链接目录
link_directories(
    # HiAI相关库目录
    ${HIAI_LIB_DIR}
)

# 源文件
set(SOURCES
    src/main.cpp
    src/inference_engine.cpp
    src/audio_processor.cpp
)

# 可执行文件
add_executable(edgevoice ${SOURCES})

# 链接库
target_link_libraries(edgevoice
    # HiAI相关库
    hiai_foundation
    neural_network_core
    # 系统库
    hilog_ndk.z
    # 其他依赖库
    m
    dl
)

# 安装规则
install(TARGETS edgevoice
    RUNTIME DESTINATION bin
)

# 打印配置信息
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}: ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}") 
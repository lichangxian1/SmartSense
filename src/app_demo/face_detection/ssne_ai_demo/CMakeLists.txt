cmake_minimum_required(VERSION 3.0.0)

project(ssne_ai_demo)

# 引入路径配置文件
include(cmake_config/Paths.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
# 核心：禁用ABI变更提示
add_compile_options(-Wno-psabi)

# ==========================================
# 1. 添加头文件目录
# ==========================================
include_directories(${M1_SDK_INC_DIR})
include_directories(${M1_SDK_INC_DIR}/smartsoc)
include_directories(${CMAKE_SOURCE_DIR}/include)

set(TARGET $ENV{BASE_DIR}/target)
message(STATUS "Build target path: " ${TARGET})

# ==========================================
# 2. 定义我们的 GCN 执行文件
# ==========================================
add_executable(ssne_ai_demo ${PROJECT_SOURCE_DIR}/demo_gcn.cpp)

# 精准添加 src 文件夹下的基础驱动文件和我们的推理文件
# (避免把官方用不到的 face 源码编译进来报错)
file(GLOB SRC_FILES 
    "${CMAKE_SOURCE_DIR}/src/pipeline_image.cpp" 
    "${CMAKE_SOURCE_DIR}/src/utils.cpp" 
    "${CMAKE_SOURCE_DIR}/src/osd-device.cpp"
    "${CMAKE_SOURCE_DIR}/src/gcn_extractor.cpp"
)
target_sources(ssne_ai_demo PRIVATE ${SRC_FILES})

# ==========================================
# 3. 链接动态库
# ==========================================
# 注意：这里必须是 ssne_ai_demo，不能是原版的 ssne_ai_demo
target_link_libraries(ssne_ai_demo 
                        ${M1_SSNE_LIB}
                        ${M1_CMABUFFER_LIB}
                        ${M1_OSD_LIB}
                        ${M1_SSZLOG_LIB}
                        ${M1_ZLOG_LIB}
                        ${M1_EMB_LIB}
)

# ==========================================
# 4. 安装/导出
# ==========================================
install(TARGETS ssne_ai_demo DESTINATION bin)
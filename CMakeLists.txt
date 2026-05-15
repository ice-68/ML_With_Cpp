cmake_minimum_required(VERSION 3.16)
project(SimpleNeuralNetwork VERSION 0.1.0 LANGUAGES CXX)

# ── C++ 标准 ───────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── GoogleTest（单元测试框架）───────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# [关键] Windows/MSVC 没有 pthread.h，GTest 会编译失败
set(gtest_force_shared_crt    ON  CACHE BOOL "" FORCE)
set(gtest_disable_pthreads    ON  CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)
enable_testing()

# ── 核心库（过滤掉 main.cpp）────────────────────
file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX ".*main\\.cpp$")

add_library(ml_core ${SOURCES})
target_include_directories(ml_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── 主程序可执行文件 ────────────────────────────
add_executable(ml_app src/main.cpp)
target_link_libraries(ml_app PRIVATE ml_core)

# ── 自动化测试 ────────────────────────────────────
file(GLOB TEST_SOURCES "tests/*.cpp")
if(TEST_SOURCES)
    add_executable(unit_tests ${TEST_SOURCES})
    target_link_libraries(unit_tests PRIVATE ml_core gtest gtest_main)
    add_test(NAME unit_tests COMMAND unit_tests)
endif()

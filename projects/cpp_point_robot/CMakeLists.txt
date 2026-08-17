cmake_minimum_required(VERSION 3.16)

project(cpp_point_robot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
)

add_library(
    point_robot_lib
    src/point_robot.cpp
)

target_include_directories(
    point_robot_lib
    PUBLIC
    include
)

add_executable(
    point_robot
    main.cpp
)

target_link_libraries(
    point_robot
    PRIVATE
    point_robot_lib
)

enable_testing()

add_executable(
    point_robot_tests
    tests/test_point_robot.cpp
)

target_link_libraries(
    point_robot_tests
    PRIVATE
    point_robot_lib
)

add_test(
    NAME point_robot_tests
    COMMAND point_robot_tests
)
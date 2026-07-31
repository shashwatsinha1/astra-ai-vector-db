cmake_minimum_required(VERSION 3.16)

project(VectorDBRealtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(vectordb main.cpp)

if (WIN32)
    target_link_libraries(vectordb PRIVATE ws2_32)
endif()

target_compile_options(vectordb PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -pedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

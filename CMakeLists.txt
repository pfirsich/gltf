if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake)
  set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")
endif()

cmake_minimum_required(VERSION 3.0.0)

project(gltf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

include(cmake/FindSimdjson.cmake)

add_compile_definitions(NOMINMAX _USE_MATH_DEFINES) # Windows is trash

include(cmake/wall.cmake)

if (GLTF_ENABLE_ASAN)
  set(GLWRAP_ENABLE_ASAN TRUE)
  include(cmake/asan.cmake)
endif()

set(SRC gltf.hpp gltf.cpp)

add_library(gltf STATIC ${SRC})
target_include_directories(gltf PUBLIC .)
target_link_libraries(gltf PRIVATE simdjson)
set_wall(gltf)

if (GLTF_BUILD_EXAMPLE)
  add_subdirectory(example)
endif()

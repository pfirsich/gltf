cmake_minimum_required(VERSION 3.0.0)

project(gltf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

include(cmake/wall.cmake)

if (GLTF_ENABLE_ASAN)
  set(GLWRAP_ENABLE_ASAN TRUE)
  include(cmake/asan.cmake)
endif()

set(SRC gltf.cpp)

list(APPEND SRC simdjson/singleheader/simdjson.cpp)
set_source_files_properties(simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_FLAGS
  "-Wno-gnu-label-as-value")

add_library(gltf STATIC ${SRC})
target_include_directories(gltf SYSTEM PRIVATE simdjson/singleheader)
set_wall(gltf)

if (GLTF_BUILD_EXAMPLE)
  add_subdirectory(example)
endif()

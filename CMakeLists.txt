cmake_minimum_required(VERSION 3.0.0)

project(gltf)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

# Enable address sanitizer in xcode so it can find the ASan library
# https://stackoverflow.com/a/53314315/5587737
set(CMAKE_XCODE_GENERATE_SCHEME ON)
set(CMAKE_XCODE_SCHEME_ADDRESS_SANITIZER ON)
set(CMAKE_XCODE_SCHEME_ADDRESS_SANITIZER_USE_AFTER_RETURN ON)

function(set_warn target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -pedantic -Werror)
  endif()
endfunction()

set(SRC gltf.cpp)

list(APPEND SRC simdjson/singleheader/simdjson.cpp)
set_source_files_properties(simdjson/singleheader/simdjson.cpp PROPERTIES COMPILE_FLAGS
  "-Wno-gnu-label-as-value")

add_subdirectory(glwrap)

add_library(gltf STATIC ${SRC})
target_include_directories(gltf SYSTEM PRIVATE simdjson/singleheader)
set_warn(gltf)

add_executable(example example.cpp)
target_include_directories(example PUBLIC glwrap/include)
target_link_libraries(example gltf glwx)
set_warn(example)

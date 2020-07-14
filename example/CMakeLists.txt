add_subdirectory(glwrap)

add_executable(example example.cpp)
target_include_directories(example PRIVATE ..)
target_include_directories(example PRIVATE glwrap/include)
target_link_libraries(example gltf glwx)
set_wall(example)

project('gltf', 'cpp', default_options: ['warning_level=3', 'cpp_std=c++17'])

vcpkg = find_program('vcpkg', required : true)
run_command(vcpkg,
  'install',
  '--x-wait-for-lock',
  '--x-manifest-root=' + meson.current_source_dir(),
  check: true)

simdjson_dep = dependency('simdjson', include_type: 'system')

gltf_deps = [simdjson_dep]
gltf = static_library('gltf', ['gltf.cpp'], dependencies: gltf_deps)
gltf_dep = declare_dependency(
  include_directories: include_directories('.'),
  dependencies: gltf_deps,
  link_with: [gltf])

if not meson.is_subproject()
  glwx_dep = dependency('glwrap', fallback: ['glwrap', 'glwx_dep'])
  executable('example', 'example/example.cpp', dependencies: [gltf_dep, glwx_dep])
endif

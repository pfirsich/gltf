# gltf
Almost single header [glTF](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0) loader library for modern C++.

# Documentation
## Basic Loading
You can use the `load` function which returns an `std::optional<Gltf>`. The optional will be empty in case of an error, in which case a log message will be outputted as well. If you want to configure logging, you can do that via the `LoadParameters::logCallback` parameter. You can also modify the behaviour of load in other ways. Please have a look at the bottom of [gltf.hpp](gltf.hpp) for more information.

## glTF Structure
This (and other glTF libraries) are not quite like e.g. [assimp](https://github.com/assimp/assimp) that you can just integrate into your project without knowing anything about the file formats you are loading.

It will mostly just fill a big struct that looks like the contents of a glTF file (+ some extra).

I will now give a basic rundown of the format. As additional material I of course recommend the [specification](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0) itself but also and especially the official cheatsheet, which is just great: [gltf 2.0 Quick Reference Guide](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0).

If that sounds too bothersome to you, you can also have a look at the example right away: [example.cpp](example/example.cpp).

While reading the following sections, you probably want to follow along in the header file: [gltf.hpp](gltf.hpp).

### Scenes (`Gltf::scenes`)
Everything you would want to use from a glTF file is part of a `Scene`, which just stores a list of `Node`-indices (`Scene::nodes`). There can be multiple scenes in a file, but an optional default scene of the file (which the implementation is supposed to use) can be present (`Gltf::scene`).

### Nodes (`Gltf::nodes`)
Nodes form the hierarchy of scene objects. Every node has a transform (`Node::transform`) and can optionally reference other components, like a camera (`Node::camera`), a skin (`Node::skin`), a mesh (`Node::mesh`) or a light (`Node::light`). It may also reference it's children (`Node::children`).

### Mesh (`Gltf::meshes`)
A mesh itself is nothing more than a list of primitives, each of which represent a single draw call.
A primitive has a list of attributes (`Mesh::Primitive::attributes`), a draw mode (`Mesh::Primitive::mode`) and may optionally reference an accessor (explained later) for indices (`Mesh::Primitive::indices`) or a material (`Mesh::Primitive::material`). If no material is given, a [default material](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#default-material) should be used.

There are predefined attribute names with specified semantics, but custom attribute names are allowed too (their names have to follow a certain scheme) ([spec](https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#meshes)). Each attribute (and `Mesh::Primitive::indices`) references an accessor, which references data).

### Buffers
This actually represents a block of data. If you load `.glb` files or just use `Buffer::uri`s with base64 data or relative paths (most likely), then the only interesting part of the Buffer will just be `Buffer::data` (`std::vector<uint8_t>`).

### BufferViews
A buffer view represents a sequence of bytes contained in a buffer. It stores a buffer reference (`BufferView::buffer`) an offset (`BufferView::byteOffset`) a length (`BufferView::byteLength`) and optionally a stride (`BufferView::byteStride`), which is the distance in bytes between consecutive elements of whatever the buffer view contains.

A buffer view may also have a `BufferView::target`, which corresponds to the binding point of your buffer view. If a buffer view has a target, you most likely want to upload it to the GPU. In OpenGL terms: all buffer views with a `target` correspond to a single VBO.

### Accessors
The accessor is another view on a buffer view, but also contains metadata about what type of data is referenced by it. It will reference a buffer view (`Accessor::bufferView`) have a type (`BufferView::type` - scalar, vec2-vec4, mat2-mat4) and a component type (`Accessor::componentType` - (unsigned) byte, (unsigned) short, unsigned int, float). The count (`BufferView::count`) will contain the number of elements (e.g. the number of `vec4`s referenced by the accessor). Integer elements may also be normalized into `[0, 1]` (`Accessor::normalized`).

`Accessor::min` and `Accessor::max` contain the maximum and minimum elements referenced by the accessor (component-wise). The number of elements is equal to the number of components the type has (e.g. 4 for vec4, 9 for mat3, etc.).

### Intermediate Summary
* Build the scene hierarchy from nodes
* Create a buffer from each buffer view with a `target`
* For each primitive in each mesh build a vertex format from the attributes

### Textures, Images, Samplers
Textures are referenced in materials as albedo (`Material::PbrMetallicRoughness::baseColorTexture`), metallic/roughness map (`Material::PbrMetallicRoughness::metallicRoughnessTexture`), normal map (`Material::normalTexture`), occlusion map (`Material::occlusionTexture`) or emissive map (`Material::emissiveTexture`) and represent only a tuple of an image (the texture data and not much more) and a sampler (the sampler parameters - min/mag filters and wrap modes).

### Animation / Skinning
TODO (complain if you want me to do this. In the meantime look at the [example](example/example.cpp))

### Camera, Light, Material
I think these are pretty self-explanatory, but if not, tell me.

# Building / Integrating
I don't really get why so many people opt for single-header instead of single header and source file, so I provide both separately.

You can just add [gltf.cpp](gltf.cpp) and [simdjson.cpp](simdjson/singleheader/simdjson.cpp) to your project (and add some include paths) or you can `add_subdirectory` this repo with CMake and link against `gltf`.

If you want to build the example, you need to define `GLTF_BUILD_EXAMPLE` as a CMake variable.

If you want to build with ASan enabled, define `GLTF_ENABLE_ASAN`.

# Why
Since other libraries don't give a quick rundown like I did above, I had to dig around in the spec anyways and after a while thought to myself that I could just load the files on my own (and of course it took weeks :D).

Also I'm a big fan of the glTF format in general and I think it's a very valuable effort, so I wanted to understand it better.

Additionally:
* Alternative libraries are either not made for C++ or not C++-ey enough for my taste (most important!).
* I wanted to try myself to make a header that documents the format as much as possible.
* I wished there was some validation built-in to make using the loaded data a little simpler (because I could assume it to be more sane).
* I didn't need image loading or saving glTF files.

# TODO
* Make the example less of an absolute mess
* Generate Normals (many of the basic [samples](https://github.com/KhronosGroup/glTF-Sample-Models) and `Fox` don't work in the example viewer because of this)
* Generate Tangents
* Morph Targets (sparse accessors, animations with `"path": "weights"`)
* Checks
  - Check that `min`/`max` are present if accessor is used for `"POSITION"`
  - Alignments
  - Valid attribute and accessor type combinations: https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#meshes
  - Check that bufferView referenced by primitive.indices has target ELEMENT_ARRAY_BUFFER, componentType must be UNSIGNED_BYTE, UNSIGNED_SHORT or UNSIGNED_INT, type must be scalar
  - Check that no `matrix` is given when node is used as animation target
* Add mesh merging? (Import the whole file as a single mesh, if possible - same material, etc.)

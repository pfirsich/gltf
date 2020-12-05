#pragma once

// We want to copy buffers as-is. According to GLTF spec they are all litte-endian
// Windows is always little endian
#ifndef _WIN32
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
#endif

#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace gltf {
using GlEnum = uint32_t; // This should be compatible with GLenum (or is supposed to)
using DisplayName = std::optional<std::string>; // non-unique optional name for top-level objects
using Data = std::vector<uint8_t>;

// Members without a member initializer are usually required

// All this json stuff is not super ergonomic, but I don't want to noise up the header
struct JsonArray;
struct JsonObject;

using JsonNull = std::monostate;
using JsonValue = std::variant<JsonNull, std::string, int64_t, uint64_t, double, bool,
    std::unique_ptr<JsonArray>, std::unique_ptr<JsonObject>>;

struct JsonArray : public std::vector<JsonValue> {
};

struct JsonObject : public std::unordered_map<std::string, JsonValue> {
};

using Extras = JsonValue;

using AccessorIndex = size_t;
using AnimationSamplerIndex = size_t;
using BufferIndex = size_t;
using BufferViewIndex = size_t;
using CameraIndex = size_t;
using ImageIndex = size_t;
using MaterialIndex = size_t;
using MeshIndex = size_t;
using NodeIndex = size_t;
using SamplerIndex = size_t;
using SceneIndex = size_t;
using SkinIndex = size_t;
using TextureIndex = size_t;
using LightIndex = size_t;

using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;
using quat = std::array<float, 4>; // x, y, z, w
using mat4 = std::array<float, 16>; // column-major order

// clang-format off
constexpr mat4 mat4Identity = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
};
// clang-format on

struct Buffer {
    // Empty if the buffer refers to the GLB BIN chunk. Only valid for buffer 0.
    std::optional<std::string> uri;
    size_t byteLength;

    DisplayName name;
    Extras extras;

    Data data;
};

struct BufferView {
    enum class Target : GlEnum {
        ArrayBuffer = 34962, // GL_ARRAY_BUFFER
        ElementArrayBuffer = 34963, // GL_ELEMENT_ARRAY_BUFFER
    };

    BufferIndex buffer; // required
    size_t byteOffset = 0;
    size_t byteLength; // required
    // byteStride must be defined if two or more accessors reference this bufferView
    std::optional<size_t> byteStride;
    std::optional<Target> target;

    DisplayName name;
    Extras extras;
};

struct Accessor {
    enum class ComponentType : GlEnum {
        Byte = 5120, // GL_BYTE
        UnsignedByte = 5121, // GL_UNSIGNED_BYTE
        Short = 5122, // GL_SHORT
        UnsignedShort = 5123, // GL_UNSIGNED_SHORT
        UnsignedInt = 5125, // GL_UNSIGNED_INT, only allowed for indices
        Float = 5126, // GL_FLOAT
    };

    enum class Type {
        Scalar = 1,
        Vec2 = 2,
        Vec3 = 3,
        Vec4 = 4,
        Mat2 = 12,
        Mat3 = 13,
        Mat4 = 14,
    };

    // If not provided accessor must be initialized with zeros
    std::optional<BufferViewIndex> bufferView;
    size_t byteOffset = 0;
    size_t count; // required, in elements, not bytes!
    ComponentType componentType; // required
    bool normalized = false;
    Type type; // required

    // In the GLTF file min and max need types that correspond to componentType,
    // but I don't want to variant them here. A double fits all values of all possible types.
    // If accessor is sparse, min/max contain values with sparse substitution applied
    // normalized has no effect on min/max values. They correspond to actual values stored
    // in the buffer
    std::vector<double> min;
    std::vector<double> max;
    // Sparse sparse;

    DisplayName name;
    Extras extras;
};

struct Camera {
    struct Perspective {
        std::optional<float> aspectRatio; // if not given, aspect of canvas should be used
        float yfov; // required
        std::optional<float> zfar; // if undefined, infinite projection must be used
        float znear; // required

        Extras extras;

        // Returns matrices as specified here:
        // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#projection-matrices
        mat4 getMatrix(float aspectRatio) const;
        mat4 getMatrix() const; // calls aspectRation.value() (throws if aspectRatio is empty)
    };

    struct Orthographic {
        float xmag; // required
        float ymag; // required
        float zfar; // required, must be greater than znear
        float znear; // required

        Extras extras;

        mat4 getMatrix() const;
    };

    // Actually this has a mandatory `type` instead
    std::variant<Orthographic, Perspective> projection;

    DisplayName name;
    Extras extras;

    mat4 getProjection(float aspectRatio) const;
};

struct Image {
    struct UriData {
        std::string uri;
        Data data;
    };

    struct BufferViewData {
        std::string mimeType; // required for BufferView data
        BufferViewIndex bufferView;
    };

    std::variant<UriData, BufferViewData> data;

    DisplayName name;
    Extras extras;

    std::pair<const uint8_t*, size_t> getData() const;
};

struct Sampler {
    enum class MinFilter : GlEnum {
        Nearest = 9728, // GL_NEAREST
        Linear = 9729, // GL_LINEAR
        NearestMipmapNearest = 9984, // GL_NEAREST_MIPMAP_NEAREST
        LinearMipmapNearest = 9985, // GL_LINEAR_MIPMAP_NEAREST
        NearestMipmapLinear = 9986, // GL_NEAREST_MIPMAP_LINEAR
        LinearMipmapLinear = 9987, // GL_LINEAR_MIPMAP_LINEAR
    };

    enum class MagFilter : GlEnum {
        Nearest = 9728,
        Linear = 9729,
    };

    enum class WrapMode {
        ClampToEdge = 33071, // GL_CLAMP_TO_EDGE
        MirroredRepeat = 33648, // GL_MIRRORED_REPEAT
        Repeat = 10497, // GL_REPEAT
    };

    std::optional<MinFilter> minFilter;
    std::optional<MagFilter> magFilter;
    WrapMode wrapS = WrapMode::Repeat;
    WrapMode wrapT = WrapMode::Repeat;

    DisplayName name;
    Extras extras;
};

struct Texture {
    // If sampler is undefined, use repeat wrapping and auto filtering
    std::optional<SamplerIndex> sampler;
    // If source is undefined and no alternate source, behaviour is undefined
    std::optional<ImageIndex> source;

    DisplayName name;
    Extras extras;
};

struct Material {
    struct TextureInfo {
        TextureIndex index; // required
        size_t texCoord = 0; // index of TEXCOORD attribute

        Extras extras;
    };

    struct NormalTextureInfo {
        TextureIndex index; // required
        size_t texCoord = 0; // index of TEXCOORD attribute
        float scale = 1.0f;

        Extras extras;
    };

    struct OcclusionTextureInfo {
        TextureIndex index; // required
        size_t texCoord = 0; // index of TEXCOORD attribute
        float strength = 1.0f;

        Extras extras;
    };

    struct PbrMetallicRoughness {
        vec4 baseColorFactor { 1.0f, 1.0f, 1.0f, 1.0f };
        std::optional<TextureInfo> baseColorTexture;
        float metallicFactor = 1.0f;
        float roughnessFactor = 1.0f;
        std::optional<TextureInfo> metallicRoughnessTexture; // metalness in B, roughness in G

        Extras extras;
    };

    enum class AlphaMode {
        Opaque,
        Mask,
        Blend,
    };

    DisplayName name;
    Extras extras;

    std::optional<PbrMetallicRoughness> pbrMetallicRoughness;
    std::optional<NormalTextureInfo> normalTexture;
    std::optional<OcclusionTextureInfo> occlusionTexture; // sampled from R channel
    std::optional<TextureInfo> emissiveTexture; // sampled from A channel
    vec3 emissiveFactor { 0.0f, 0.0f, 0.0f };
    AlphaMode alphaMode = AlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
};

struct Mesh {
    struct Primitive {
        struct Attribute {
            std::string id;
            AccessorIndex accessor;
        };

        enum class Mode : GlEnum {
            Points = 0, // GL_POINTS
            Lines = 1, // GL_LINES
            LineLoop = 2, // GL_LINE_LOPP
            LineStrip = 3, // GL_LINE_STRIP
            Triangles = 4, // GL_TRIANGLES
            TriangleStrip = 5, // GL_TRIANGLE_STRIP
            TriangleFan = 6, // GL_TRIANGLE_FAN
        };

        std::vector<Attribute> attributes;
        std::optional<AccessorIndex> indices;
        std::optional<MaterialIndex> material;
        Mode mode = Mode::Triangles;
        // std::vector<Target> targets

        Extras extras;
    };

    std::vector<Primitive> primitives;
    // std::vector<float> weights;

    DisplayName name;
    Extras extras;
};

struct Skin {
    std::optional<AccessorIndex> inverseBindMatrices;
    // Root node, but not really necessary and may give weird results if used (CesiumMan).
    // I found it better to just use the node that has the skin attached as the root node.
    std::optional<NodeIndex> skeleton;
    std::vector<NodeIndex> joints; // required

    DisplayName name;
    Extras extras;
};

struct Node {
    struct Trs {
        vec3 translation { 0.0f, 0.0f, 0.0f };
        vec3 scale { 1.0f, 1.0f, 1.0f };
        quat rotation { 0.0f, 0.0f, 0.0f, 1.0f };

        mat4 getMatrix() const;
    };

    // NOTE: (from spec:) "When matrix is defined, it must be decomposable to TRS. This implies that
    // transformation matrices cannot skew or shear."
    using Transform = std::variant<mat4, Trs>;

    std::optional<CameraIndex> camera;
    std::vector<NodeIndex> children; // values are unique
    std::optional<SkinIndex> skin;
    Transform transform = mat4Identity;
    std::optional<MeshIndex> mesh;
    // std::vector<float> weights;

    DisplayName name;
    Extras extras;

    std::optional<LightIndex> light; // KHR_lights_punctual

    // this parent field is not part of the spec! it's just for convenience
    // and if a parent exists, it is well defined, because (spec):
    // "no node may be a direct descendant of more than one node"
    std::optional<NodeIndex> parent;

    mat4 getTransformMatrix() const;
};

struct Scene {
    std::vector<NodeIndex> nodes;

    DisplayName name;
    Extras extras;
};

struct Animation {
    struct Channel {
        struct Target {
            enum class Path {
                Translation,
                Rotation,
                Scale,
                Weights,
            };

            std::optional<NodeIndex> node;
            Path path; // required, should this be a string?

            Extras extras;
        };

        AnimationSamplerIndex sampler; // required
        Target target; // required

        Extras extras;
    };

    struct Sampler {
        enum class Interpolation {
            Linear,
            Step,
            Cubicspline,
        };

        // Accessor is scalar float. Values are sequence of times >= 0, strictly increasing
        AccessorIndex input; // required
        Interpolation interpolation = Interpolation::Linear;
        AccessorIndex output; // required

        Extras extras;
    };

    std::vector<Channel> channels; // required
    std::vector<Sampler> samplers; // required

    DisplayName name;
    Extras extras;
};

// https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_lights_punctual
struct Light {
    struct Directional {
    };

    // recommendation for range:
    // attenuation = max(min(1.0 - (current_distance / range)^4, 1), 0) / current_distance^2

    struct Point {
        std::optional<float> range; // infinite if undefined
    };

    struct Spot {
        std::optional<float> range; // infinite if undefined
        float innerConeAngle = 0.0f;
        float outerConeAngle = M_PI_4;
    };

    DisplayName name;
    vec3 color { 1.0f, 1.0f, 1.0f }; // linear RGB
    float intensity = 1.0f; // point & spot: candela, directional: lux
    std::variant<Directional, Point, Spot> parameters;
};

struct Gltf {
    struct Asset {
        std::optional<std::string> copyright;
        std::optional<std::string> generator;
        std::string version; // required
        std::optional<std::string> minVersion;

        Extras extras;
    };

    std::vector<std::string> extensionsUsed;
    std::vector<std::string> extensionsRequired;

    std::vector<Accessor> accessors;
    std::vector<Animation> animations;
    Asset asset; // required
    std::vector<Buffer> buffers;
    std::vector<BufferView> bufferViews;
    std::vector<Camera> cameras;
    std::vector<Image> images;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    std::vector<Node> nodes;
    std::vector<Sampler> samplers;
    std::optional<SceneIndex> scene;
    std::vector<Scene> scenes;
    std::vector<Skin> skins;
    std::vector<Texture> textures;

    Extras extras;

    std::vector<Light> lights; // KHR_lights_punctual

    std::pair<const uint8_t*, size_t> getBufferViewData(BufferViewIndex idx) const;
    std::pair<const uint8_t*, size_t> getImageData(ImageIndex idx) const;
    std::pair<const uint8_t*, size_t> getAccessorData(AccessorIndex idx) const;
};

enum class LogSeverity { Warning, Error };
// Warnings will be emitted if the file is malformed in any way.
// Errors will be emitted if the parsing fails.
using LogCallback = void(LogSeverity severity, std::string_view message);

using BufferLoader = std::optional<std::vector<uint8_t>>(std::string_view uri);

// Also here so you can use it for your loaders
std::optional<std::vector<uint8_t>> loadDataUri(std::string_view dataUri);

// This is here so you can use it for your buffer loaders, if you need to
std::function<BufferLoader> makeDefaultBufferLoader(const std::filesystem::path& baseDir);

struct LoadParameters {
    std::function<LogCallback> logCallback = nullptr;
    std::function<BufferLoader> bufferLoader = nullptr;
    bool validateIndices = true;
    bool setParents = true;
    bool guessMissingBufferViewTarget = true;
};

// NOTE: I thought about providing a stream-IO callbacks, but I think it will only be
// useful for the JSON loading and that will be pretty quick (comparatively) already.
// The buffers will have to be read completely into a vector<uint8_t> anyways, so there won't
// be much use for streaming IO either there either.
// If you want to use streaming IO to load your textures for example, pass a buffer loader
// that will just do nothing instead and load them yourself later.
// For GLB files buffer 0 (the BIN chunk) will be loaded either way, so if you want to avoid
// that as well, use a separate .bin file.

// The two load functions distinguish .gltf or .glb by themselves

// This will use the current directory as the base directory for loading buffers from relative path
// uris
std::optional<Gltf> load(const uint8_t* data, size_t size, const LoadParameters& parameters = {});

// This will the use the containing directory of the path argument as the base directory for buffers
// from relative path uris
std::optional<Gltf> load(const std::filesystem::path& path, const LoadParameters& parameters = {});
}

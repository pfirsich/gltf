#include "gltf.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <unordered_set>

#include <simdjson.h>

using namespace std::literals;

namespace gltf {
mat4 Camera::Perspective::getMatrix(float aspectRatio) const
{
    const auto invTanY = 1.0f / std::tan(0.5f * yfov);
    mat4 ret {};
    ret[0] = invTanY / aspectRatio;
    ret[5] = invTanY;
    ret[11] = -1.0f;

    if (zfar) {
        const auto n = znear, f = *zfar;
        ret[10] = (n + f) / (n - f);
        ret[14] = 2.0f * n * f / (n - f);
    } else {
        ret[10] = -1.0f;
        ret[14] = -2.0f * znear;
    }
    return ret;
}

mat4 Camera::Perspective::getMatrix() const
{
    return getMatrix(aspectRatio.value());
}

mat4 Camera::Orthographic::getMatrix() const
{
    mat4 ret {};
    ret[0] = 1.0f / xmag;
    ret[5] = 1.0f / ymag;
    ret[10] = 2.0f / (znear - zfar);
    ret[14] = (znear + zfar) / (znear - zfar);
    ret[15] = 1.0f;
    return ret;
}

mat4 Camera::getProjection(float aspectRatio) const
{
    if (std::holds_alternative<Camera::Orthographic>(projection))
        return std::get<Camera::Orthographic>(projection).getMatrix();
    else
        return std::get<Camera::Perspective>(projection).getMatrix(aspectRatio);
}

mat4 Node::Trs::getMatrix() const
{
    const auto& t = translation;
    const auto& q = rotation;
    const auto& s = scale;

    // Mostly from glm
    float qxx = q[0] * q[0];
    float qyy = q[1] * q[1];
    float qzz = q[2] * q[2];
    float qxz = q[0] * q[2];
    float qxy = q[0] * q[1];
    float qyz = q[1] * q[2];
    float qwx = q[3] * q[0];
    float qwy = q[3] * q[1];
    float qwz = q[3] * q[2];

    float r0 = 1.0f - 2.0f * (qyy + qzz);
    float r1 = 2.0f * (qxy + qwz);
    float r2 = 2.0f * (qxz - qwy);

    float r3 = 2.0f * (qxy - qwz);
    float r4 = 1.0f - 2.0f * (qxx + qzz);
    float r5 = 2.0f * (qyz + qwx);

    float r6 = 2.0f * (qxz + qwy);
    float r7 = 2.0f * (qyz - qwx);
    float r8 = 1.0f - 2.0f * (qxx + qyy);

    mat4 m;
    // clang-format off
    m[0] = s[0] * r0; m[4] = s[1] * r3; m[ 8] = s[2] * r6; m[12] = t[0];
    m[1] = s[0] * r1; m[5] = s[1] * r4; m[ 9] = s[2] * r7; m[13] = t[1];
    m[2] = s[0] * r2; m[6] = s[1] * r5; m[10] = s[2] * r8; m[14] = t[2];
    m[3] =      0.0f; m[7] =      0.0f; m[11] =      0.0f; m[15] = 1.0f;
    // clang-format on
    return m;
}

mat4 Node::getTransformMatrix() const
{
    if (std::holds_alternative<mat4>(transform))
        return std::get<mat4>(transform);
    else
        return std::get<Trs>(transform).getMatrix();
}

std::pair<const uint8_t*, size_t> Gltf::getBufferViewData(BufferViewIndex idx) const
{
    const auto& bufferView = bufferViews[idx];
    const auto& data = buffers[bufferView.buffer].data;
    assert(bufferView.byteOffset < data.size());
    assert(bufferView.byteOffset + bufferView.byteLength <= data.size());
    return std::make_pair(data.data() + bufferView.byteOffset, bufferView.byteLength);
}

std::pair<const uint8_t*, size_t> Gltf::getImageData(ImageIndex idx) const
{
    const auto imageData = &images[idx].data;
    if (const auto uriData = std::get_if<Image::UriData>(imageData)) {
        return std::make_pair(uriData->data.data(), uriData->data.size());
    } else if (const auto bvData = std::get_if<Image::BufferViewData>(imageData)) {
        return getBufferViewData(bvData->bufferView);
    }
    std::abort();
}

std::pair<const uint8_t*, size_t> Gltf::getAccessorData(AccessorIndex idx) const
{
    const auto& accessor = accessors[idx];
    assert(accessor.bufferView);
    const auto& bufferView = bufferViews[*accessor.bufferView];
    const auto data = buffers[bufferView.buffer].data.data();
    assert(accessor.byteOffset < bufferView.byteLength);
    return std::make_pair(data + bufferView.byteOffset + accessor.byteOffset,
        bufferView.byteLength - accessor.byteOffset);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

constexpr std::array<std::string_view, 1> supportedExtensions { "KHR_lights_punctual" };

std::unordered_map<LogSeverity, std::string_view> severityToString {
    { LogSeverity::Warning, "Warning" },
    { LogSeverity::Error, "Error" },
};

template <typename... Args>
std::string fmt(Args&&... args)
{
    std::stringstream ss;
    (ss << ... << args);
    return ss.str();
}

struct Logger {
    const std::function<LogCallback>& logCallback;

    template <typename... Args>
    void log(LogSeverity severity, Args&&... args) const
    {
        if (logCallback) {
            logCallback(severity, fmt(std::forward<Args>(args)...));
        } else {
            std::cerr << "glTF Loader [" << severityToString.at(severity) << "]: ";
            (std::cerr << ... << args);
            std::cerr << "\n";
        }
    }

    template <typename... Args>
    void warn(Args&&... args) const
    {
        log(LogSeverity::Warning, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(Args&&... args) const
    {
        log(LogSeverity::Error, std::forward<Args>(args)...);
    }
};

constexpr size_t jsonPadding = simdjson::SIMDJSON_PADDING;

constexpr uint32_t glbMagic = 0x46546C67; // ascii: "glTF"
constexpr uint32_t glbVersion = 2;
constexpr uint32_t jsonChunkType = 0x4E4F534A; // ascii: "JSON"
constexpr uint32_t binChunkType = 0x004E4942; // ascii: "BIN"

struct GlbHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t length;
};

struct GlbChunkHeader {
    uint32_t length;
    uint32_t type;
};

class JsonException : public std::exception {
public:
    JsonException(const std::string& message)
        : message(message)
    {
    }

    JsonException(std::string_view path, std::string_view type)
    {
        message = fmt("\"", path, "\" must be a ", type);
    }

    const char* what() const throw()
    {
        return message.c_str();
    }

private:
    std::string message;
};

template <typename... Args>
void error(Args&&... args)
{
    throw JsonException(fmt(std::forward<Args>(args)...));
}

template <typename... Args>
void parseAssert(bool cond, Args&&... args)
{
    if (!cond)
        error(std::forward<Args>(args)...);
}

void unknownKey(std::string_view key, std::string_view path)
{
    error("Unknown property \"", key, "\" in \"", path, "\"");
}

simdjson::dom::parser& getSimdJsonParser()
{
    static simdjson::dom::parser parser;
    return parser;
}

template <typename T, typename GetType = T>
T get(std::string_view type, const simdjson::dom::element& element, std::string_view path)
{
    const auto [v, error] = element.get<GetType>();
    if (error)
        throw JsonException(path, type);

    if constexpr (std::is_same_v<T, GetType>)
        return v;
    else
        return static_cast<T>(v);
}

template <typename T>
T get(const simdjson::dom::element& element, std::string_view path);

template <>
std::string get<std::string>(const simdjson::dom::element& element, std::string_view path)
{
    return get<std::string, std::string_view>("string", element, path);
}

template <>
simdjson::dom::object get<simdjson::dom::object>(
    const simdjson::dom::element& element, std::string_view path)
{
    return get<simdjson::dom::object>("object", element, path);
}

template <>
simdjson::dom::array get<simdjson::dom::array>(
    const simdjson::dom::element& element, std::string_view path)
{
    return get<simdjson::dom::array>("array", element, path);
}

template <>
uint64_t get<uint64_t>(const simdjson::dom::element& element, std::string_view path)
{
    return get<uint64_t>("unsigned integer", element, path);
}

template <>
float get<float>(const simdjson::dom::element& element, std::string_view path)
{
    return get<float, double>("float", element, path);
}

template <>
double get<double>(const simdjson::dom::element& element, std::string_view path)
{
    return get<double, double>("double", element, path);
}

template <>
bool get<bool>(const simdjson::dom::element& element, std::string_view path)
{
    return get<bool>("bool", element, path);
}

template <typename T>
void getMath(T& obj, size_t num, const simdjson::dom::element& element, std::string_view path,
    bool normalized = false)
{
    const auto arr = get<simdjson::dom::array>(element, path);
    const auto ptr = &obj[0];
    size_t i = 0;
    const std::string elemName = std::string(path) + "[]";
    for (const auto item : arr) {
        if (i >= num)
            error("Array is too large for \"", path, "\" (should be ", num, ")");
        // vec3, vec4, quat, mat4 are all in the same order as gltf
        ptr[i] = get<float>(item, elemName);
        if (normalized)
            if (ptr[i] < 0.0f || ptr[i] > 1.0f)
                error("Array elements have to be in [0, 1] for \"", path, "\"");
        i++;
    }
    if (i < num)
        error("Array is too small for \"", path, "\" (should be ", num, ")");
}

void readJsonValue(JsonValue& value, const simdjson::dom::element& element)
{
    switch (element.type()) {
    case simdjson::dom::element_type::NULL_VALUE:
        value = JsonNull {};
        break;
    case simdjson::dom::element_type::STRING:
        value = std::string(std::string_view(element));
        break;
    case simdjson::dom::element_type::INT64:
        value = int64_t(element);
        break;
    case simdjson::dom::element_type::UINT64:
        value = uint64_t(element);
        break;
    case simdjson::dom::element_type::DOUBLE:
        value = double(element);
        break;
    case simdjson::dom::element_type::BOOL:
        value = bool(element);
        break;
    case simdjson::dom::element_type::ARRAY: {
        value = std::make_unique<JsonArray>();
        auto& array = *std::get<std::unique_ptr<JsonArray>>(value);
        for (const auto v : simdjson::dom::array(element))
            readJsonValue(array.emplace_back(), v);
    } break;
    case simdjson::dom::element_type::OBJECT: {
        value = std::make_unique<JsonObject>();
        auto& object = *std::get<std::unique_ptr<JsonObject>>(value);
        for (const auto [k, v] : simdjson::dom::object(element))
            readJsonValue(object.emplace(k, JsonNull {}).first->second, v);
    } break;
    default:
        assert(false && "Unknown JSON type");
    }
}

void readExtras(Extras& extras, const simdjson::dom::element& element)
{
    readJsonValue(extras, element);
}

void readAsset(Gltf& file, const simdjson::dom::element& elem, const Logger& logger)
{
    using namespace simdjson;

    const auto obj = get<dom::object>(elem, "asset");

    bool foundVersion = false;
    for (const auto [key, value] : obj) {
        if (key == "copyright") {
            file.asset.copyright = get<std::string>(value, "asset.copyright");
        } else if (key == "generator") {
            file.asset.generator = get<std::string>(value, "asset.generator");
        } else if (key == "version") {
            file.asset.version = get<std::string>(value, "asset.version");
            foundVersion = true;
        } else if (key == "minVersion") {
            file.asset.minVersion = get<std::string>(value, "asset.minVersion");
        } else if (key == "extensions") {
            // Ignore unknown extensions
        } else if (key == "extras") {
            readExtras(file.asset.extras, value);
        } else {
            unknownKey(key, "asset");
        }
    }

    if (!foundVersion) {
        // I should probably error, but at least I want to try to keep going
        logger.warn("Missing \"asset.version\" property.");
    }
}

void readScenes(Gltf& file, const simdjson::dom::element& elem, const Logger& /*logger*/)
{
    using namespace simdjson;

    const auto scenes = get<dom::array>(elem, "scenes");
    for (const auto item : scenes) {
        const auto obj = get<dom::object>(item, "scenes[]");
        auto& scene = file.scenes.emplace_back();
        for (const auto [key, value] : obj) {
            if (key == "nodes") {
                const auto arr = get<dom::array>(value, "scenes[].nodes");
                for (const auto item : arr)
                    scene.nodes.push_back(get<uint64_t>(item, "scenes[].nodes[]"));
            } else if (key == "name") {
                scene.name = get<std::string>(value, "scenes[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(scene.extras, value);
            } else {
                unknownKey(key, "scenes[]");
            }
        }
    }
}

void readSkins(Gltf& file, const simdjson::dom::element& elem, const Logger& /*logger*/)
{
    using namespace simdjson;

    const auto skins = get<dom::array>(elem, "skins");
    for (const auto item : skins) {
        const auto obj = get<dom::object>(item, "skins[]");
        auto& skin = file.skins.emplace_back();
        bool foundJoints = false;
        for (const auto [key, value] : obj) {
            if (key == "inverseBindMatrices") {
                skin.inverseBindMatrices = get<uint64_t>(value, "skins[].inverseBindMatrices");
            } else if (key == "skeleton") {
                skin.skeleton = get<uint64_t>(value, "skins[].inverseBindMatrices");
            } else if (key == "joints") {
                const auto arr = get<dom::array>(value, "skins[].joints");
                for (const auto item : arr)
                    skin.joints.push_back(get<uint64_t>(item, "skins[].joints[]"));
                foundJoints = true;
            } else if (key == "name") {
                skin.name = get<std::string>(value, "skins[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(skin.extras, value);
            } else {
                unknownKey(key, "skins[]");
            }
        }
        parseAssert(foundJoints, "Missing mandatory \"joints\" in skin");
    }
}

void readTextures(Gltf& file, const simdjson::dom::element& elem, const Logger& /*logger*/)
{
    using namespace simdjson;

    const auto textures = get<dom::array>(elem, "textures");
    for (const auto item : textures) {
        const auto obj = get<dom::object>(item, "textures[]");
        auto& texture = file.textures.emplace_back();
        for (const auto [key, value] : obj) {
            if (key == "sampler") {
                texture.sampler = get<uint64_t>(value, "textures[].sampler");
            } else if (key == "source") {
                texture.source = get<uint64_t>(value, "textures[].source");
            } else if (key == "name") {
                texture.name = get<std::string>(value, "textures[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(texture.extras, value);
            } else {
                unknownKey(key, "textures[]");
            }
        }
    }
}

void readNodes(Gltf& file, const simdjson::dom::element& elem, const Logger& logger)
{
    using namespace simdjson;

    const auto nodes = get<dom::array>(elem, "nodes");
    for (const auto item : nodes) {
        const auto obj = get<dom::object>(item, "nodes[]");
        auto& node = file.nodes.emplace_back();
        bool foundTrs = false, foundMatrix = false;
        for (const auto [key, value] : obj) {
            if (key == "camera") {
                node.camera = get<uint64_t>(value, "nodes[].camera");
            } else if (key == "children") {
                const auto arr = get<dom::array>(value, "nodes[].children");
                for (const auto item : arr)
                    node.children.push_back(get<uint64_t>(item, "nodes[].children[]"));
            } else if (key == "skin") {
                node.skin = get<uint64_t>(value, "nodes[].skin");
            } else if (key == "matrix") {
                mat4 mat;
                getMath<mat4>(mat, 16, value, "nodes[].matrix");
                node.transform = mat;
                foundMatrix = true;
            } else if (key == "translation") {
                if (!foundMatrix) {
                    if (std::holds_alternative<mat4>(node.transform))
                        node.transform = Node::Trs {};
                    auto& translation = std::get<Node::Trs>(node.transform).translation;
                    getMath<vec3>(translation, 3, value, "nodes[].translation");
                }
                foundTrs = true;
            } else if (key == "scale") {
                if (!foundMatrix) {
                    if (std::holds_alternative<mat4>(node.transform))
                        node.transform = Node::Trs {};
                    auto& scale = std::get<Node::Trs>(node.transform).scale;
                    getMath<vec3>(scale, 3, value, "nodes[].scale");
                }
                foundTrs = true;
            } else if (key == "rotation") {
                if (!foundMatrix) {
                    if (std::holds_alternative<mat4>(node.transform))
                        node.transform = Node::Trs {};
                    auto& rotation = std::get<Node::Trs>(node.transform).rotation;
                    getMath<quat>(rotation, 4, value, "nodes[].rotation");
                }
                foundTrs = true;
            } else if (key == "mesh") {
                node.mesh = get<uint64_t>(value, "nodes[].mesh");
            } else if (key == "weights") {
                logger.warn("Morph targets are not implemented yet.");
            } else if (key == "name") {
                node.name = get<std::string>(value, "nodes[].name");
            } else if (key == "extensions") {
                const auto obj = get<dom::object>(value, "nodes[].extensions");
                for (const auto [key, value] : obj) {
                    if (key == "KHR_lights_punctual") {
                        const auto obj
                            = get<dom::object>(value, "nodes[].extensions.KHR_lights_punctual");
                        for (const auto [key, value] : obj) {
                            if (key == "light") {
                                node.light = get<uint64_t>(
                                    value, "nodes[].extensions.KHR_lights_punctual.light");
                            } else {
                                unknownKey(key, "nodes[].extensions.KHR_lights_punctual");
                            }
                        }
                    }
                    // Ignore unknown extensions
                }
            } else if (key == "extras") {
                readExtras(node.extras, value);
            } else {
                unknownKey(key, "nodes[]");
            }
        }
        parseAssert(!foundTrs || !foundMatrix, "Node ", file.nodes.size() - 1,
            " has both matrix and TRS properties defined");
    }
}

void readSamplers(Gltf& file, const simdjson::dom::element& elem, const Logger& /*logger*/)
{
    using namespace simdjson;

    const auto samplers = get<dom::array>(elem, "samplers");
    for (const auto item : samplers) {
        const auto obj = get<dom::object>(item, "samplers[]");
        auto& sampler = file.samplers.emplace_back();
        for (const auto [key, value] : obj) {
            if (key == "minFilter") {
                sampler.minFilter
                    = static_cast<Sampler::MinFilter>(get<uint64_t>(value, "samplers[].minFilter"));
            } else if (key == "magFilter") {
                sampler.minFilter
                    = static_cast<Sampler::MinFilter>(get<uint64_t>(value, "samplers[].minFilter"));
            } else if (key == "wrapS") {
                sampler.wrapS
                    = static_cast<Sampler::WrapMode>(get<uint64_t>(value, "samplers[].wrapS"));
            } else if (key == "wrapT") {
                sampler.wrapT
                    = static_cast<Sampler::WrapMode>(get<uint64_t>(value, "samplers[].wrapT"));
            } else if (key == "name") {
                sampler.name = get<std::string>(value, "samplers[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(sampler.extras, value);
            } else {
                unknownKey(key, "samplers[]");
            }
        }
    }
}

size_t getComponentCount(Accessor::Type type)
{
    switch (type) {
    case Accessor::Type::Scalar:
        return 1;
    case Accessor::Type::Vec2:
        return 2;
    case Accessor::Type::Vec3:
        return 3;
    case Accessor::Type::Vec4:
        return 4;
    case Accessor::Type::Mat2:
        return 4;
    case Accessor::Type::Mat3:
        return 9;
    case Accessor::Type::Mat4:
        return 16;
    default:
        return 0;
    }
}

Accessor::Type parseAccessorType(std::string_view type)
{
    if (type == "SCALAR")
        return Accessor::Type::Scalar;
    else if (type == "VEC2")
        return Accessor::Type::Vec2;
    else if (type == "VEC3")
        return Accessor::Type::Vec3;
    else if (type == "VEC4")
        return Accessor::Type::Vec4;
    else if (type == "MAT2")
        return Accessor::Type::Mat2;
    else if (type == "MAT3")
        return Accessor::Type::Mat3;
    else if (type == "MAT4")
        return Accessor::Type::Mat4;
    else
        error("Invalid attribute type");
    return Accessor::Type {}; // Silence stupid warning
}

void readAccessors(Gltf& file, const simdjson::dom::element& elem, const Logger& logger)
{
    using namespace simdjson;

    const auto accessors = get<dom::array>(elem, "accessors");
    for (const auto item : accessors) {
        const auto obj = get<dom::object>(item, "accessors[]");
        auto& accessor = file.accessors.emplace_back();
        bool foundCount = false, foundComponentType = false, foundType = false;
        for (const auto [key, value] : obj) {
            if (key == "bufferView") {
                accessor.bufferView = get<uint64_t>(value, "accessors[].bufferView");
            } else if (key == "byteOffset") {
                accessor.byteOffset = get<uint64_t>(value, "accessors[].byteOffset");
            } else if (key == "count") {
                accessor.count = get<uint64_t>(value, "accessors[].count");
                foundCount = true;
            } else if (key == "componentType") {
                accessor.componentType = static_cast<Accessor::ComponentType>(
                    get<uint64_t>(value, "accessors[].componentType"));
                foundComponentType = true;
            } else if (key == "normalized") {
                accessor.normalized = get<bool>(value, "accessors[].normalized");
            } else if (key == "type") {
                accessor.type = parseAccessorType(get<std::string>(value, "accessors[].type"));
                foundType = true;
            } else if (key == "min") {
                const auto arr = get<simdjson::dom::array>(value, "accessors[].min");
                for (const auto val : arr)
                    accessor.min.push_back(get<double>(val, "accessors[].min[]"));
            } else if (key == "max") {
                const auto arr = get<simdjson::dom::array>(value, "accessors[].max");
                for (const auto val : arr)
                    accessor.max.push_back(get<double>(val, "accessors[].max[]"));
            } else if (key == "sparse") {
                logger.warn("Sparse accessors are not implemented yet.");
            } else if (key == "name") {
                accessor.name = get<std::string>(value, "accessors[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(accessor.extras, value);
            } else {
                unknownKey(key, "accessors[]");
            }
        }

        parseAssert(foundCount, "Missing mandatory \"count\" in accessor");
        parseAssert(foundComponentType, "Missing mandatory \"componentType\" in accessor");
        parseAssert(foundType, "Missing mandatory \"type\" in accessor");

        const auto compCount = getComponentCount(accessor.type);
        if (!accessor.min.empty())
            parseAssert(accessor.min.size() == compCount, "\"min\" must have size ", compCount,
                " to fit \"type\"");
        if (!accessor.max.empty())
            parseAssert(accessor.max.size() == compCount, "\"max\" must have size ", compCount,
                " to fit \"type\"");
    }
}

Animation::Channel::Target::Path parseChannelPath(std::string_view path)
{
    if (path == "translation")
        return Animation::Channel::Target::Path::Translation;
    else if (path == "rotation")
        return Animation::Channel::Target::Path::Rotation;
    else if (path == "scale")
        return Animation::Channel::Target::Path::Scale;
    else if (path == "weights")
        return Animation::Channel::Target::Path::Weights;
    else
        error("Invalid channel path \"", path, "\"");
    return Animation::Channel::Target::Path {};
}

Animation::Sampler::Interpolation parseSamplerInterpolation(std::string_view interpolation)
{
    if (interpolation == "LINEAR")
        return Animation::Sampler::Interpolation::Linear;
    else if (interpolation == "STEP")
        return Animation::Sampler::Interpolation::Step;
    else if (interpolation == "CUBICSPLINE")
        return Animation::Sampler::Interpolation::Cubicspline;
    else
        error("Invalid sampler interpolation \"", interpolation, "\"");
    return Animation::Sampler::Interpolation {};
}

void readAnimations(Gltf& file, const simdjson::dom::element& elem, const Logger& /*logger*/)
{
    using namespace simdjson;

    const auto animations = get<dom::array>(elem, "animations");
    for (const auto item : animations) {
        const auto obj = get<dom::object>(item, "animations[]");
        auto& animation = file.animations.emplace_back();
        bool foundChannels = false, foundSamplers = false;
        for (const auto [key, value] : obj) {
            if (key == "channels") {
                const auto arr = get<dom::array>(value, "animations[].channels");
                for (const auto item : arr) {
                    const auto obj = get<dom::object>(item, "animations[].channels[]");
                    auto& channel = animation.channels.emplace_back();
                    bool foundSampler = false, foundTarget = false;
                    for (const auto [key, value] : obj) {
                        if (key == "sampler") {
                            channel.sampler
                                = get<uint64_t>(value, "animations[].channels[].sampler");
                            foundSampler = true;
                        } else if (key == "target") {
                            const auto obj
                                = get<dom::object>(value, "animations[].channels[].target");
                            for (const auto [key, value] : obj) {
                                if (key == "node") {
                                    channel.target.node = get<uint64_t>(
                                        value, "animations[].channels[].target.node");
                                } else if (key == "path") {
                                    channel.target.path = parseChannelPath(get<std::string>(
                                        value, "animations[].channels[].target.path"));
                                } else if (key == "extensions") {
                                    // Ignore unknown extensions
                                } else if (key == "extras") {
                                    readExtras(channel.target.extras, value);
                                } else {
                                    unknownKey(key, "animations[].channels[].target");
                                }
                            }
                            foundTarget = true;
                        } else if (key == "extensions") {
                            // Ignore unknown extensions
                        } else if (key == "extras") {
                            readExtras(channel.extras, value);
                        } else {
                            unknownKey(key, "animations[].channels[]");
                        }
                    }
                    parseAssert(foundSampler, "Missing mandatory \"sampler\" in channel");
                    parseAssert(foundTarget, "Missing mandatory \"target\" in channel");
                }
                foundChannels = true;
            } else if (key == "samplers") {
                const auto arr = get<dom::array>(value, "animations[].samplers");
                for (const auto item : arr) {
                    const auto obj = get<dom::object>(item, "animations[].samplers[]");
                    auto& sampler = animation.samplers.emplace_back();
                    bool foundInput = false, foundOutput = false;
                    for (const auto [key, value] : obj) {
                        if (key == "input") {
                            sampler.input = get<uint64_t>(value, "animations[].samplers[].input");
                            foundInput = true;
                        } else if (key == "interpolation") {
                            sampler.interpolation = parseSamplerInterpolation(
                                get<std::string>(value, "animations[].samplers[].interpolation"));
                        } else if (key == "output") {
                            sampler.output = get<uint64_t>(value, "animations[].samplers[].output");
                            foundOutput = true;
                        } else if (key == "extensions") {
                            // Ignore unknown extensions
                        } else if (key == "extras") {
                            readExtras(sampler.extras, value);
                        } else {
                            unknownKey(key, "animations[].samplers[]");
                        }
                    }
                    parseAssert(foundInput, "Missing mandatory \"input\" in sampler");
                    parseAssert(foundOutput, "Missing mandatory \"output\" in sampler");
                }
                foundSamplers = true;
            } else if (key == "name") {
                animation.name = get<std::string>(value, "animations[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(animation.extras, value);
            } else {
                unknownKey(key, "animations[]");
            }
        }
        parseAssert(foundChannels, "Missing mandatory \"channels\" in animation");
        parseAssert(foundSamplers, "Missing mandatory \"samplers\" in animation");
    }
}

void readBuffers(Gltf& file, const simdjson::dom::element& elem, const Logger& /*logger*/)
{
    using namespace simdjson;

    const auto buffers = get<dom::array>(elem, "buffers");
    for (const auto item : buffers) {
        const auto obj = get<dom::object>(item, "buffers[]");
        auto& buffer = file.buffers.emplace_back();
        bool foundByteLength = false;
        for (const auto [key, value] : obj) {
            if (key == "uri") {
                buffer.uri = get<std::string>(value, "buffers[].uri");
            } else if (key == "byteLength") {
                buffer.byteLength = get<uint64_t>(value, "buffers[].byteLength");
                foundByteLength = true;
            } else if (key == "name") {
                buffer.name = get<std::string>(value, "buffer[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(buffer.extras, value);
            } else {
                unknownKey(key, "buffer[]");
            }
        }
        parseAssert(foundByteLength, "Missing mandatory \"byteLength\" in buffers");

        // Buffers will be filled after the JSON parsing is done
        if (!buffer.uri && file.buffers.size() > 1)
            error("Buffer ", file.buffers.size() - 1,
                " has undefined \"uri\" but cannot refer to BIN chunk");
    }
}

void readBufferViews(Gltf& file, const simdjson::dom::element& elem, const Logger& /*logger*/)
{
    using namespace simdjson;

    const auto bufferViews = get<dom::array>(elem, "bufferViews");
    for (const auto item : bufferViews) {
        const auto obj = get<dom::object>(item, "bufferViews[]");
        auto& bufferView = file.bufferViews.emplace_back();
        bool foundBuffer = false, foundByteLength = false;
        for (const auto [key, value] : obj) {
            if (key == "buffer") {
                bufferView.buffer = get<uint64_t>(value, "bufferViews[].buffer");
                foundBuffer = true;
            } else if (key == "byteOffset") {
                bufferView.byteOffset = get<uint64_t>(value, "bufferViews[].byteOffset");
            } else if (key == "byteLength") {
                bufferView.byteLength = get<uint64_t>(value, "bufferViews[].byteLength");
                foundByteLength = true;
            } else if (key == "byteStride") {
                bufferView.byteStride = get<uint64_t>(value, "bufferViews[].byteStride");
            } else if (key == "target") {
                bufferView.target
                    = static_cast<BufferView::Target>(get<uint64_t>(value, "bufferViews[].target"));
            } else if (key == "name") {
                bufferView.name = get<std::string>(value, "bufferView[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(bufferView.extras, value);
            } else {
                unknownKey(key, "bufferView[]");
            }
        }

        parseAssert(foundBuffer, "Missing mandatory \"buffer\" in bufferViews");
        parseAssert(foundByteLength, "Missing mandatory \"byteLength\" in bufferViews");
    }
}

void readCameras(Gltf& file, const simdjson::dom::element& elem, const Logger& /*logger*/)
{
    using namespace simdjson;

    const auto cameras = get<dom::array>(elem, "cameras");
    for (const auto item : cameras) {
        const auto obj = get<dom::object>(item, "cameras[]");
        std::optional<Camera::Perspective> perspective;
        std::optional<Camera::Orthographic> orthographic;
        std::string type;
        auto& camera = file.cameras.emplace_back();
        for (const auto [key, value] : obj) {
            if (key == "type") {
                type = get<std::string>(value, "cameras[].type");
                if (type != "perspective" && type != "orthographic")
                    error("Invalid value for \"type\" in camera");
            } else if (key == "perspective") {
                const auto obj = get<dom::object>(value, "cameras[].perspective");
                perspective = Camera::Perspective {};
                bool foundYfov = false, foundZnear = false;
                for (const auto [key, value] : obj) {
                    if (key == "aspectRatio") {
                        perspective->aspectRatio
                            = get<float>(value, "cameras[].perspective.aspectRatio");
                    } else if (key == "yfov") {
                        perspective->yfov = get<float>(value, "cameras[].perspective.yfov");
                        foundYfov = true;
                    } else if (key == "zfar") {
                        perspective->zfar = get<float>(value, "cameras[].perspective.zfar");
                    } else if (key == "znear") {
                        perspective->znear = get<float>(value, "cameras[].perspective.znear");
                        foundZnear = true;
                    } else if (key == "extensions") {
                        // Ignore unknown extensions
                    } else if (key == "extras") {
                        readExtras(perspective->extras, value);
                    }
                }
                parseAssert(foundYfov, "Missing mandatory \"yfov\" in camera.perspective");
                parseAssert(foundZnear, "Missing mandatory \"znear\" in camera.perspective");
            } else if (key == "orthographic") {
                const auto obj = get<dom::object>(value, "cameras[].orthographic");
                orthographic = Camera::Orthographic {};
                bool foundXmag = false, foundYMag = false, foundZfar = false, foundZnear = false;
                for (const auto [key, value] : obj) {
                    if (key == "xmag") {
                        orthographic->xmag = get<float>(value, "cameras[].orthographic.xmag");
                        foundXmag = true;
                    } else if (key == "ymag") {
                        orthographic->ymag = get<float>(value, "cameras[].orthographic.ymag");
                        foundYMag = true;
                    } else if (key == "zfar") {
                        orthographic->zfar = get<float>(value, "cameras[].orthographic.zfar");
                        foundZfar = true;
                    } else if (key == "znear") {
                        orthographic->znear = get<float>(value, "cameras[].orthographic.znear");
                        foundZnear = true;
                    } else if (key == "extensions") {
                        // Ignore unknown extensions
                    } else if (key == "extras") {
                        readExtras(orthographic->extras, value);
                    }
                }
                parseAssert(foundXmag, "Missing mandatory \"xmag\" in camera.orthographic");
                parseAssert(foundYMag, "Missing mandatory \"ymag\" in camera.orthographic");
                parseAssert(foundZfar, "Missing mandatory \"zfar\" in camera.orthographic");
                parseAssert(foundZnear, "Missing mandatory \"znear\" in camera.orthographic");
            } else if (key == "name") {
                camera.name = get<std::string>(value, "cameras[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(camera.extras, value);
            } else {
                unknownKey(key, "cameras[]");
            }
        }

        parseAssert(!type.empty(), "Missing mandatory \"type\" in camera");
        if (type == "perspective") {
            parseAssert(perspective.has_value(),
                "Missing \"perspective\" in camera even though type is \"perspective\"");
            camera.projection = std::move(*perspective);
        } else if (type == "orthographic") {
            parseAssert(orthographic.has_value(),
                "Missing \"orthographic\" in camera even though type is \"orthographic\"");
            camera.projection = std::move(*orthographic);
        } else {
            assert(false);
        }
    }
}

void readImages(Gltf& file, const simdjson::dom::element& elem, const Logger& /*logger*/)
{
    using namespace simdjson;

    const auto images = get<dom::array>(elem, "images");
    for (const auto item : images) {
        const auto obj = get<dom::object>(item, "images[]");
        std::optional<std::string> uri;
        std::optional<BufferViewIndex> bufferView;
        std::optional<std::string> mimeType;
        auto& image = file.images.emplace_back();
        for (const auto [key, value] : obj) {
            if (key == "uri") {
                uri = get<std::string>(value, "images[].uri");
            } else if (key == "mimeType") {
                mimeType = get<std::string>(value, "images[].mimeType");
            } else if (key == "bufferView") {
                bufferView = get<uint64_t>(value, "images[].bufferView");
            } else if (key == "name") {
                image.name = get<std::string>(value, "images[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(image.extras, value);
            } else {
                unknownKey(key, "images[]");
            }
        }

        parseAssert(
            uri || bufferView, "Either \"uri\" or \"bufferView\" must be defined for image");
        parseAssert(
            !uri || !bufferView, "Only one of \"uri\" or \"bufferView\" must be defined for image");

        if (uri) {
            image.data = Image::UriData { *uri, {} };
            // The data will be loaded later
        } else if (bufferView) {
            parseAssert(mimeType.has_value(),
                "\"mimeType\" is mandatory if bufferView is defined for image");
            image.data = Image::BufferViewData { *mimeType, *bufferView };
        } else {
            assert(false && "Either uri or bufferView must be true");
        }
    }
}

Material::AlphaMode parseAlphaMode(std::string_view mode)
{
    if (mode == "OPAQUE")
        return Material::AlphaMode::Opaque;
    else if (mode == "MASK")
        return Material::AlphaMode::Mask;
    else if (mode == "BLEND")
        return Material::AlphaMode::Blend;
    else
        error("Invalid alphaMode type");
    return Material::AlphaMode {}; // Silence stupid warning
}

template <typename T>
void readTextureInfo(T& textureInfo, const simdjson::dom::element& elem, std::string_view path)
{
    using namespace simdjson;

    const auto obj = get<dom::object>(elem, path);
    bool foundIndex = false;
    for (const auto [key, value] : obj) {
        if (key == "index") {
            textureInfo.index = get<uint64_t>(value, std::string(path) + ".index");
            foundIndex = true;
        } else if (key == "texCoord") {
            textureInfo.texCoord = get<uint64_t>(value, std::string(path) + ".texCoord");
        } else if (key == "scale") {
            if constexpr (std::is_same_v<T, Material::NormalTextureInfo>) {
                textureInfo.scale = get<float>(value, std::string(path) + ".scale");
            } else {
                unknownKey(key, path);
            }
        } else if (key == "strength") {
            if constexpr (std::is_same_v<T, Material::OcclusionTextureInfo>) {
                textureInfo.strength = get<float>(value, std::string(path) + ".strength");
            } else {
                unknownKey(key, path);
            }
        } else if (key == "extensions") {
            // Ignore unknown extensions
        } else if (key == "extras") {
            readExtras(textureInfo.extras, value);
        } else {
            unknownKey(key, path);
        }
    }
    parseAssert(foundIndex, "\"index\" is mandatory in \"", path, "\"");
}

void readMaterials(Gltf& file, const simdjson::dom::element& elem, const Logger& /*logger*/)
{
    using namespace simdjson;

    const auto materials = get<dom::array>(elem, "materials");
    for (const auto item : materials) {
        const auto obj = get<dom::object>(item, "materials[]");
        auto& material = file.materials.emplace_back();
        for (const auto [key, value] : obj) {
            if (key == "name") {
                material.name = get<std::string>(value, "materials[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(material.extras, value);
            } else if (key == "pbrMetallicRoughness") {
                const auto obj = get<dom::object>(value, "materials[].pbrMetallicRoughness");
                material.pbrMetallicRoughness = Material::PbrMetallicRoughness {};
                auto& pbr = *material.pbrMetallicRoughness;
                for (const auto [key, value] : obj) {
                    if (key == "baseColorFactor") {
                        getMath<vec4>(pbr.baseColorFactor, 4, value,
                            "materials[].pbrMetallicRoughness.baseColorFactor", true);
                    } else if (key == "baseColorTexture") {
                        pbr.baseColorTexture = Material::TextureInfo {};
                        readTextureInfo(*pbr.baseColorTexture, value,
                            "materials[].pbrMetallicRoughness.baseColorTexture");
                    } else if (key == "metallicFactor") {
                        pbr.metallicFactor
                            = get<float>(value, "materials[].pbrMetallicRoughness.metallicFactor");
                    } else if (key == "roughnessFactor") {
                        pbr.roughnessFactor
                            = get<float>(value, "materials[].pbrMetallicRoughness.roughnessFactor");
                    } else if (key == "metallicRoughnessTexture") {
                        pbr.metallicRoughnessTexture = Material::TextureInfo {};
                        readTextureInfo(*pbr.metallicRoughnessTexture, value,
                            "materials[].pbrMetallicRoughness.metallicRoughnessTexture");
                    } else if (key == "extensions") {
                        // Ignore unknown extensions
                    } else if (key == "extras") {
                        readExtras(pbr.extras, value);
                    }
                }
            } else if (key == "normalTexture") {
                material.normalTexture = Material::NormalTextureInfo {};
                readTextureInfo(*material.normalTexture, value, "materials[].normalTexture");
            } else if (key == "occlusionTexture") {
                material.occlusionTexture = Material::OcclusionTextureInfo {};
                readTextureInfo(*material.occlusionTexture, value, "materials[].occlusionTexture");
            } else if (key == "emissiveTexture") {
                material.emissiveTexture = Material::TextureInfo {};
                readTextureInfo(*material.emissiveTexture, value, "materials[].emissiveTexture");
            } else if (key == "emissiveFactor") {
                getMath<vec3>(
                    material.emissiveFactor, 3, value, "materials[].emissiveFactor", true);
            } else if (key == "alphaMode") {
                const auto mode = get<std::string>(value, "materials[].alphaMode");
                material.alphaMode = parseAlphaMode(mode);
            } else if (key == "alphaCutoff") {
                material.alphaCutoff = get<float>(value, "materials[].alphaCutoff");
            } else if (key == "doubleSided") {
                material.doubleSided = get<bool>(value, "materials[].doubleSided");
            } else {
                unknownKey(key, "materials[]");
            }
        }
    }
}

void checkAttributeName(std::string_view name)
{
    static constexpr std::array<std::string_view, 4> indexedAttributes { "TEXCOORD_", "COLOR_",
        "JOINTS_", "WEIGHTS_" };
    if (name.size() == 0)
        error("Attribute name length is 0");

    if (name[0] == '_') // application specific, anything goes
        return;

    if (name == "POSITION" || name == "NORMAL" || name == "TANGENT")
        return;

    for (const auto attr : indexedAttributes) {
        if (name.find(attr) == 0) {
            const auto index = name.substr(attr.size());
            if (index.empty() || index.find_first_not_of("0123456789") != std::string::npos)
                error("Invalid attribute index for attribute \"", name, "\"");
            else
                return;
        }
    }

    error("Invalid attribute name \"", name, "\"");
}

void readMeshes(Gltf& file, const simdjson::dom::element& elem, const Logger& logger)
{
    using namespace simdjson;

    const auto meshes = get<dom::array>(elem, "meshes");
    for (const auto item : meshes) {
        const auto obj = get<dom::object>(item, "meshes[]");
        auto& mesh = file.meshes.emplace_back();
        for (const auto [key, value] : obj) {
            if (key == "primitives") {
                const auto arr = get<dom::array>(value, "meshes[].primitives");
                for (const auto item : arr) {
                    const auto obj = get<dom::object>(item, "meshes[].primitives[]");
                    auto& primitive = mesh.primitives.emplace_back();
                    for (const auto [key, value] : obj) {
                        if (key == "attributes") {
                            const auto obj
                                = get<dom::object>(value, "meshes[].primitives[].attributes");
                            for (const auto [key, value] : obj) {
                                checkAttributeName(key);
                                // I usually don't do this, but this goes too far
                                // clang-format off
                                primitive.attributes.push_back( Mesh::Primitive::Attribute {
                                    std::string(key),
                                    get<uint64_t>(value, "meshes[].primitives[].attributes[]")
                                });
                                // clang-format on
                            }
                        } else if (key == "indices") {
                            primitive.indices
                                = get<uint64_t>(value, "meshes[].primitives[].indices");
                        } else if (key == "material") {
                            primitive.material
                                = get<uint64_t>(value, "meshes[].primitives[].material");
                        } else if (key == "mode") {
                            primitive.mode = static_cast<Mesh::Primitive::Mode>(
                                get<uint64_t>(value, "meshes[].primitives[].mode"));
                        } else if (key == "targets") {
                            logger.warn("Morph targets are not implemented yet.");
                        } else if (key == "extensions") {
                            // Ignore unknown extensions
                        } else if (key == "extras") {
                            readExtras(primitive.extras, value);
                        } else {
                            unknownKey(key, "meshes[].primitives[]");
                        }
                    }
                }
            } else if (key == "weights") {
                logger.warn("Morph targets are not implemented yet.");
            } else if (key == "name") {
                mesh.name = get<std::string>(value, "meshes[].name");
            } else if (key == "extensions") {
                // Ignore unknown extensions
            } else if (key == "extras") {
                readExtras(mesh.extras, value);
            } else {
                unknownKey(key, "meshes[]");
            }
        }
    }
}

void readLights(Gltf& file, const simdjson::dom::element& elem, const Logger& /*logger*/)
{
    using namespace simdjson;

    const auto lights = get<dom::array>(elem, "lights");
    for (const auto item : lights) {
        const auto obj = get<dom::object>(item, "lights[]");
        std::optional<Light::Spot> spot;
        std::optional<std::string> type;
        std::optional<float> range;
        auto& light = file.lights.emplace_back();
        for (const auto [key, value] : obj) {
            if (key == "name") {
                light.name
                    = get<std::string>(value, "extensions.KHR_lights_punctual.lights[].name");
            } else if (key == "color") {
                getMath<vec3>(
                    light.color, 3, value, "extensions.KHR_lights_punctual.lights[].color", true);
            } else if (key == "intensity") {
                light.intensity
                    = get<float>(value, "extensions.KHR_lights_punctual.lights[].intensity");
            } else if (key == "type") {
                type = get<std::string>(
                    value, "extensions.KHR_lights_punctual.lights[].intensity.type");
                if (*type != "directional" && *type != "point" && *type != "spot")
                    error("Invalid value for \"type\" in light");
            } else if (key == "range") {
                range = get<float>(value, "extensions.KHR_lights_punctual.lights[].range");
            } else if (key == "spot") {
                const auto obj
                    = get<dom::object>(value, "extensions.KHR_lights_punctual.lights[].spot");
                spot = Light::Spot {};
                for (const auto [key, value] : obj) {
                    if (key == "innerConeAngle") {
                        spot->innerConeAngle = get<float>(
                            value, "extensions.KHR_lights_punctual.lights[].spot.innerConeAngle");
                    } else if (key == "outerConeAngle") {
                        spot->outerConeAngle = get<float>(
                            value, "extensions.KHR_lights_punctual.lights[].spot.outerConeAngle");
                    } else {
                        unknownKey(key, "extensions.KHR_lights_punctual.lights[].spot");
                    }
                }
            } else {
                unknownKey(key, "extensions.KHR_lights_punctual.lights[]");
            }
        }

        parseAssert(type.has_value(), "Missing mandatory \"type\" in light");

        if (spot.has_value() && (*type == "directional" || *type == "point")) {
            error("Property \"spot\" defined for light with type \"", *type, "\"");
        }

        if (*type == "directional") {
            parseAssert(!range.has_value(),
                "Property \"range\" defined for light with type \"directional\"");
            light.parameters = Light::Directional {};
        } else if (*type == "point") {
            light.parameters = Light::Point { range };
        } else if (*type == "spot") {
            light.parameters = *spot;
            std::get<Light::Spot>(light.parameters).range = range;
        } else {
            assert(false);
        }
    }
}

void validateIndices(const Gltf& file)
{
    auto checkOob = [](std::string_view path, size_t index, size_t size) {
        if (index > size)
            error("\"", path, "\" is out of bounds: ", index, " > ", size);
    };

    for (const auto& accessor : file.accessors)
        if (accessor.bufferView)
            checkOob("accessors[].bufferView", *accessor.bufferView, file.bufferViews.size());

    for (const auto& anim : file.animations) {
        for (const auto& ch : anim.channels) {
            checkOob("animations[].channels[].sampler", ch.sampler, anim.samplers.size());
            if (ch.target.node)
                checkOob("animations[].channels[].target.node", *ch.target.node, file.nodes.size());
        }

        for (const auto& sampler : anim.samplers) {
            checkOob("animations[].samplers[].input", sampler.input, file.accessors.size());
            checkOob("animations[].samplers[].output", sampler.output, file.accessors.size());
        }
    }

    // Nothing to do for buffers

    for (const auto& bufferView : file.bufferViews)
        checkOob("bufferViews[].buffer", bufferView.buffer, file.buffers.size());

    // Nothing to do for cameras

    for (const auto& image : file.images) {
        if (const auto bvData = std::get_if<Image::BufferViewData>(&image.data))
            checkOob("images[].bufferView", bvData->bufferView, file.bufferViews.size());
    }

    for (const auto& material : file.materials) {
        if (material.pbrMetallicRoughness) {
            const auto& pbr = *material.pbrMetallicRoughness;
            if (pbr.baseColorTexture)
                checkOob("materials[].pbrMetallicRoughness.baseColorTexture.index",
                    pbr.baseColorTexture->index, file.textures.size());
            if (pbr.metallicRoughnessTexture)
                checkOob("materials[].pbrMetallicRoughness.metallicRoughnessTexture.index",
                    pbr.metallicRoughnessTexture->index, file.textures.size());
        }

        if (material.normalTexture)
            checkOob("materials[].normalTexture.index", material.normalTexture->index,
                file.textures.size());

        if (material.occlusionTexture)
            checkOob("materials[].occlusionTexture.index", material.occlusionTexture->index,
                file.textures.size());

        if (material.emissiveTexture)
            checkOob("materials[].emissiveTexture.index", material.emissiveTexture->index,
                file.textures.size());
    }

    for (const auto& mesh : file.meshes) {
        for (const auto& primitive : mesh.primitives) {
            for (const auto& attribute : primitive.attributes)
                checkOob("meshes[].primitives[].attributes[].accessor[]", attribute.accessor,
                    file.accessors.size());

            if (primitive.indices)
                checkOob(
                    "meshes[].primitives[].indices", *primitive.indices, file.accessors.size());

            if (primitive.material)
                checkOob(
                    "meshes[].primitives[].material", *primitive.material, file.materials.size());
        }
    }

    for (const auto& node : file.nodes) {
        if (node.camera)
            checkOob("nodes[].camera", *node.camera, file.cameras.size());

        std::set<NodeIndex> uniqueChildren(node.children.begin(), node.children.end());
        if (uniqueChildren.size() != node.children.size())
            error("nodes[].children are not unique");

        if (node.mesh)
            checkOob("nodes[].mesh", *node.mesh, file.meshes.size());

        if (node.skin)
            checkOob("nodes[].skin", *node.skin, file.skins.size());

        if (node.light)
            checkOob("nodes[].light", *node.light, file.lights.size());
    }

    // Nothing to do for samplers

    for (const auto& scene : file.scenes)
        for (const auto node : scene.nodes)
            checkOob("scenes[].nodes", node, file.nodes.size());

    for (const auto& skin : file.skins) {
        if (skin.inverseBindMatrices)
            checkOob(
                "skins[].inverseBindMatrices", *skin.inverseBindMatrices, file.accessors.size());

        if (skin.skeleton)
            checkOob("skins[].skeleton", *skin.skeleton, file.nodes.size());

        for (const auto node : skin.joints)
            checkOob("skins[].joints[]", node, file.nodes.size());
    }

    for (const auto& texture : file.textures) {
        if (texture.sampler)
            checkOob("textures[].sampler", *texture.sampler, file.samplers.size());

        if (texture.source)
            checkOob("textures[].source", *texture.source, file.textures.size());
    }

    if (file.scene)
        checkOob("scene", *file.scene, file.scenes.size());
}

void setParents(Gltf& file)
{
    for (size_t i = 0; i < file.nodes.size(); ++i) {
        for (const auto child : file.nodes[i].children) {
            if (file.nodes[child].parent) {
                error("Node ", child, " has multiple parents");
            }
            file.nodes[child].parent = i;
        }
    }

    for (const auto& scene : file.scenes) {
        for (const auto rootNode : scene.nodes) {
            parseAssert(!file.nodes[rootNode].parent,
                "Nodes referenced in scene must be root nodes (no parent)");
        }
    }
}

std::optional<BufferView::Target> guessBufferViewTarget(
    const Gltf& file, BufferViewIndex bufferView)
{
    for (const auto& mesh : file.meshes) {
        for (const auto& prim : mesh.primitives) {
            if (prim.indices && file.accessors[*prim.indices].bufferView == bufferView)
                return BufferView::Target::ElementArrayBuffer;

            for (const auto& attr : prim.attributes) {
                if (file.accessors[attr.accessor].bufferView == bufferView)
                    return BufferView::Target::ArrayBuffer;
            }
        }
    }
    return std::nullopt;
}

void guessMissingBufferViewTarget(Gltf& file)
{
    for (size_t i = 0; i < file.bufferViews.size(); ++i) {
        auto& bufferView = file.bufferViews[i];
        if (!bufferView.target)
            bufferView.target = guessBufferViewTarget(file, i);
    }
}

namespace base64 {
    namespace detail {
        // +1 for zero terminator of string literal
        static constexpr std::array<char, 64 + 1> base64EncodeTable {
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        };

        constexpr std::array<uint8_t, 256> decodeTable()
        {
            std::array<uint8_t, 256> table { 0 };
            for (uint8_t i = 0; i < base64EncodeTable.size(); ++i)
                table[base64EncodeTable[i]] = i;
            return table;
        }

        constexpr bool isValidCharacter(char c)
        {
            return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                || c == '+' || c == '/';
        }
    }

    std::vector<uint8_t> decode(std::string_view data)
    {
        static constexpr auto tbl = detail::decodeTable();

        std::vector<uint8_t> decoded;
        // This is too big, if there are invalid characters (will be ignored), but that's fine
        decoded.reserve((data.size() / 4) * 3);

        size_t i = 0; // only increments for every base64 character
        char last = 0;
        for (const auto c : data) {
            if (detail::isValidCharacter(c)) {
                switch (i % 4) {
                case 0:
                    // We need another character to do anything
                    break;
                case 1:
                    decoded.push_back(((tbl[last] & 0b111111) << 2) | ((tbl[c] & 0b110000) >> 4));
                    break;
                case 2:
                    decoded.push_back(((tbl[last] & 0b001111) << 4) | ((tbl[c] & 0b111100) >> 2));
                    break;
                case 3:
                    decoded.push_back(((tbl[last] & 0b000011) << 6) | ((tbl[c] & 0b111111) >> 0));
                    break;
                default:
                    break;
                }
                ++i;
                last = c;
            }
        }
        return decoded;
    }
}

std::optional<std::vector<uint8_t>> loadDataUri(std::string_view dataUri)
{
    if (dataUri.substr(0, 5) != "data:") {
        error("Invalid data uri \"", dataUri, "\": no \"data\" prefix");
        return std::nullopt;
    }

    const auto base64 = ";base64,"sv;
    const auto pos = dataUri.find(base64);
    if (pos == std::string_view::npos) {
        error("Only base64 data URIs are supported");
        return std::nullopt;
    }

    const auto data = dataUri.substr(pos + base64.size());
    return base64::decode(data);
}

std::optional<std::vector<uint8_t>> defaultBufferLoader(
    std::string_view uri, const std::filesystem::path& baseDir)
{
    if (uri.substr(0, 5) == "data:")
        return loadDataUri(uri);

    const auto path = baseDir / uri;
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        error("Could not open file \"", path, "\"");
        return std::nullopt;
    }

    file.seekg(0, std::ios::end);
    const size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size, 0);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        error("Could not read file \"", path, "\"");
        return std::nullopt;
    }
    return data;
}

std::function<BufferLoader> makeDefaultBufferLoader(const std::filesystem::path& baseDir)
{
    return [baseDir](std::string_view uri) { return defaultBufferLoader(uri, baseDir); };
}

void checkVersion(const Gltf::Asset& asset, const Logger& logger)
{
    // exactly the version we are targeting. we are fine
    if (asset.version == "2.0")
        return;

    // version does not match, but minVersion is small enough
    // if there was a version before the version we are targeting (2.0), we would have to do
    // a range check here
    if (asset.minVersion) {
        if (*asset.minVersion == "2.0")
            return;
        else
            error("\"minVersion\" is too high");
    }

    // Version does not match and no minVersion given => check major version
    const auto pos = asset.version.find('.');
    if (pos == std::string::npos)
        error("Malformed \"version\"");

    const auto majorStr = asset.version.substr(0, pos + 1);
    if (majorStr == "2") {
        // Version does not match exactly, but glTF is forwards-compatible, so it should work
        logger.warn("\"asset.version\" is not 2.0");
        return;
    }

    error("Unsupported asset version");
}

void checkRequiredExtensions(const std::vector<std::string>& required)
{
    if (required.empty())
        return;

    std::unordered_set<std::string_view> unsupported(required.cbegin(), required.cend());

    for (const auto ext : supportedExtensions)
        unsupported.erase(ext);

    if (unsupported.size() > 0) {
        std::stringstream ss;
        ss << "[";
        bool first = true;
        for (const auto ext : unsupported) {
            ss << (!first ? ", " : "") << ext;
            first = false;
        }
        ss << "]";
        error("Required extensions are not supported: ", ss.str());
    }
}

std::optional<Gltf> loadJson(const uint8_t* buffer, size_t size, bool padded,
    std::function<BufferLoader> bufferLoader, const Logger& logger,
    const LoadParameters& parameters)
{
    using namespace simdjson;

    auto& parser = getSimdJsonParser();
    dom::element json;
    error_code err;
    parser.parse(buffer, size, !padded).tie(json, err);
    if (err) {
        logger.error("Error parsing JSON document: ", err);
        return std::nullopt;
    }

    try {
        dom::object jsonObj;
        json.get<dom::object>().tie(jsonObj, err);
        if (err)
            error("Invalid JSON document");

        Gltf file;
        bool foundAsset = false;
        for (const auto [key, value] : jsonObj) {
            if (key == "asset") {
                readAsset(file, value, logger);
                foundAsset = true;
                checkVersion(file.asset, logger);
            } else if (key == "extensionsUsed") {
                const auto arr = get<simdjson::dom::array>(value, "extensionsUsed");
                for (const auto v : arr)
                    file.extensionsUsed.push_back(get<std::string>(v, "extensionsUsed[]"));

                std::unordered_set<std::string_view> supported(
                    supportedExtensions.cbegin(), supportedExtensions.cend());
                for (const auto& ext : file.extensionsUsed) {
                    if (supported.count(ext) == 0)
                        logger.warn("File uses unsupported extension \"", ext, "\"");
                }
            } else if (key == "extensionsRequired") {
                const auto arr = get<simdjson::dom::array>(value, "extensionsRequired");
                for (const auto v : arr)
                    file.extensionsRequired.push_back(get<std::string>(v, "extensionsRequired[]"));
                checkRequiredExtensions(file.extensionsRequired);
            } else if (key == "accessors") {
                readAccessors(file, value, logger);
            } else if (key == "animations") {
                readAnimations(file, value, logger);
            } else if (key == "buffers") {
                readBuffers(file, value, logger);
            } else if (key == "bufferViews") {
                readBufferViews(file, value, logger);
            } else if (key == "cameras") {
                readCameras(file, value, logger);
            } else if (key == "images") {
                readImages(file, value, logger);
            } else if (key == "materials") {
                readMaterials(file, value, logger);
            } else if (key == "meshes") {
                readMeshes(file, value, logger);
            } else if (key == "nodes") {
                readNodes(file, value, logger);
            } else if (key == "samplers") {
                readSamplers(file, value, logger);
            } else if (key == "scene") {
                file.scene = get<uint64_t>(value, "scene");
            } else if (key == "scenes") {
                readScenes(file, value, logger);
            } else if (key == "skins") {
                readSkins(file, value, logger);
            } else if (key == "textures") {
                readTextures(file, value, logger);
            } else if (key == "extensions") {
                const auto obj = get<dom::object>(value, "extensions");
                for (const auto [key, value] : obj) {
                    if (key == "KHR_lights_punctual") {
                        const auto obj = get<dom::object>(value, "extensions.KHR_lights_punctual");
                        for (const auto [key, value] : obj) {
                            if (key == "lights") {
                                readLights(file, value, logger);
                            } else {
                                unknownKey(key, "extensions.KHR_lights_punctual");
                            }
                        }
                    }
                    // Ignore unknown extensions
                }
            } else if (key == "extras") {
                readExtras(file.extras, value);
            } else {
                error("Unknown property \"", key, "\" in glTF file");
            }
        }
        parseAssert(foundAsset, "Missing \"asset\" property");

        for (auto& buffer : file.buffers) {
            if (buffer.uri) {
                auto data = bufferLoader(*buffer.uri);
                if (!data)
                    return std::nullopt; // already logged something
                buffer.data = std::move(*data);
            }
        }

        for (auto& image : file.images) {
            if (auto uriData = std::get_if<Image::UriData>(&image.data)) {
                auto data = bufferLoader(uriData->uri);
                if (!data)
                    return std::nullopt; // already logged something
                uriData->data = std::move(*data);
            }
        }

        if (parameters.validateIndices)
            validateIndices(file);
        if (parameters.setParents)
            setParents(file);
        if (parameters.guessMissingBufferViewTarget)
            guessMissingBufferViewTarget(file);

        return file;
    } catch (const JsonException& jsonExc) {
        logger.error(jsonExc.what());
        return std::nullopt;
    }
}

bool checkGlbHeader(const GlbHeader& header, size_t size, const Logger& logger)
{
    if (header.version != glbVersion) {
        logger.error("Invalid GLB version. Only version 2 is supported.");
        return false;
    }

    if (size < header.length) {
        logger.error("File is too small. Length in header: ", header.length);
        return false;
    }

    return true;
}

bool checkJsonChunkHeader(const GlbChunkHeader& chunkHeader, const Logger& logger)
{
    if (chunkHeader.type != jsonChunkType) {
        logger.error("Unexpected chunk type. First chunk must be JSON");
        return false;
    }
    return true;
}

bool checkGlbBinChunk(const GlbChunkHeader& binChunkHeader, const Buffer& buffer0, Logger& logger)
{
    if (binChunkHeader.type != binChunkType) {
        logger.error("Unexpected chunk type. Second chunk must be BIN chunk.");
        return false;
    }

    if (binChunkHeader.length < buffer0.byteLength) {
        logger.error("BIN chunk too small");
        return false;
    }

    const auto binChunkSizeDiff = binChunkHeader.length - buffer0.byteLength;
    if (binChunkSizeDiff > 3) { // max 3 bytes of padding
        logger.error("BIN chunk too large");
        return false;
    }

    return true;
}

std::optional<Gltf> load(const uint8_t* data, size_t size, const LoadParameters& parameters)
{
    Logger logger { parameters.logCallback };

    std::function<BufferLoader> bufferLoader = parameters.bufferLoader
        ? parameters.bufferLoader
        : makeDefaultBufferLoader(std::filesystem::current_path());

    const auto glbHeaderSize = sizeof(GlbHeader) + sizeof(GlbChunkHeader);
    if (size < glbHeaderSize || *reinterpret_cast<const uint32_t*>(data) != glbMagic) {
        // We assume it's JSON
        return loadJson(data, size, false, bufferLoader, logger, parameters);
    } else {
        // We assume it's GLB
        const auto& glbHeader = *reinterpret_cast<const GlbHeader*>(data);
        if (!checkGlbHeader(glbHeader, size, logger))
            return std::nullopt; // already logged something

        const auto chunkPtr = data + sizeof(GlbHeader);
        const auto& chunkHeader = *reinterpret_cast<const GlbChunkHeader*>(chunkPtr);

        if (!checkJsonChunkHeader(chunkHeader, logger))
            return std::nullopt; // already logged something

        auto chunkDataPtr = data + sizeof(GlbHeader) + sizeof(GlbChunkHeader);
        auto gltfFile
            = loadJson(chunkDataPtr, chunkHeader.length, false, bufferLoader, logger, parameters);
        if (!gltfFile)
            return std::nullopt; // already logged something

        // The first buffer refers to the BIN chunk
        if (!gltfFile->buffers.empty() && !gltfFile->buffers[0].uri) {
            auto binChunkPtr = chunkDataPtr + chunkHeader.length;
            const auto& binChunkHeader = *reinterpret_cast<const GlbChunkHeader*>(binChunkPtr);

            auto& buffer0 = gltfFile->buffers[0];
            if (!checkGlbBinChunk(binChunkHeader, buffer0, logger))
                return std::nullopt; // already logged something

            auto binChunkDataPtr = binChunkPtr + sizeof(GlbChunkHeader);
            buffer0.data.insert(
                buffer0.data.end(), binChunkDataPtr, binChunkDataPtr + binChunkHeader.length);
        }

        return gltfFile;
    }
}

std::string toHexStream(const uint8_t* buffer, size_t size)
{
    std::stringstream ss;
    for (size_t i = 0; i < size; ++i) {
        ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(buffer[i]);
        if (i < size - 1)
            ss << " ";
    }
    return ss.str();
}

std::optional<Gltf> load(const std::filesystem::path& path, const LoadParameters& parameters)
{
    Logger logger { parameters.logCallback };

    std::function<BufferLoader> bufferLoader = parameters.bufferLoader
        ? parameters.bufferLoader
        : makeDefaultBufferLoader(path.parent_path());

    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        logger.error("Could not open file");
        return std::nullopt;
    }

    file.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(file.tellg());
    file.seekg(0);

    auto read = [&file](auto ptr, size_t count) -> bool {
        return static_cast<bool>(file.read(reinterpret_cast<char*>(ptr), count));
    };

    GlbHeader glbHeader;
    if (!read(&glbHeader, sizeof(GlbHeader))) {
        logger.error("Could not read header");
        return std::nullopt;
    }

    if (file.gcount() < static_cast<long>(sizeof(uint32_t)) || glbHeader.magic != glbMagic) {
        // It's probably JSON
        // Seek to 0 and read the whole file
        // Hopefully seeking back 4 bytes is pretty much free
        file.seekg(0);
        std::vector<uint8_t> contents(size + jsonPadding);
        if (!read(contents.data(), size)) {
            logger.error("Could not read JSON file");
            return std::nullopt;
        }
        return loadJson(contents.data(), size, true, bufferLoader, logger, parameters);
    } else {
        // We assume it's GLB
        if (!checkGlbHeader(glbHeader, size, logger))
            return std::nullopt; // already logged something

        GlbChunkHeader chunkHeader;
        if (!read(&chunkHeader, sizeof(GlbChunkHeader))) {
            logger.error("Could not read json chunk header");
            return std::nullopt;
        }

        if (!checkJsonChunkHeader(chunkHeader, logger))
            return std::nullopt;

        // Read the whole JSON chunk
        std::vector<uint8_t> jsonData(chunkHeader.length + jsonPadding);
        if (!read(jsonData.data(), chunkHeader.length)) {
            logger.error("Could not read JSON chunk");
            return std::nullopt;
        }

        auto gltfFile
            = loadJson(jsonData.data(), chunkHeader.length, true, bufferLoader, logger, parameters);
        if (!gltfFile)
            return std::nullopt; // already logged something

        // The first buffer refers to the BIN chunk
        if (!gltfFile->buffers.empty() && !gltfFile->buffers[0].uri) {
            GlbChunkHeader binChunkHeader;
            if (!read(&binChunkHeader, sizeof(GlbChunkHeader))) {
                logger.error("Could not read BIN chunk header");
                return std::nullopt;
            }

            auto& buffer0 = gltfFile->buffers[0];
            if (!checkGlbBinChunk(binChunkHeader, buffer0, logger))
                return std::nullopt; // already logged something

            // We complicated the loading logic here, so we can load directly into the buffer
            buffer0.data.resize(buffer0.byteLength, 0);
            if (!read(buffer0.data.data(), buffer0.byteLength)) {
                logger.error("Could not read BIN chunk data");
                return std::nullopt;
            }
        }

        return gltfFile;
    }
}
}

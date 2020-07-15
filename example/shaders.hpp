#include <string_view>

using namespace std::literals;

namespace AttributeLocations {
constexpr size_t Position = 0;
constexpr size_t Normal = 1;
constexpr size_t Tangent = 2;
constexpr size_t TexCoord0 = 3;
constexpr size_t TexCoord1 = 4;
constexpr size_t Color0 = 5;
constexpr size_t Joints0 = 6;
constexpr size_t Weights0 = 7;
}

const auto vert = R"(
    #version 330 core

    uniform mat4 modelMatrix;
    uniform mat4 viewMatrix;
    uniform mat4 projectionMatrix;
    uniform mat3 normalMatrix;

    layout (location = 0) in vec3 attrPosition;
    layout (location = 1) in vec3 attrNormal;
    layout (location = 3) in vec2 attrTexCoords;

    out vec2 texCoords;
    out vec3 normal; // view space

    void main() {
        texCoords = attrTexCoords;
        normal = normalMatrix * attrNormal;
        gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(attrPosition, 1.0);
    }
)"sv;

const auto skinningVert = R"(
    #version 330 core

    const int MAX_JOINTS = 32;

    uniform mat4 modelMatrix;
    uniform mat4 viewMatrix;
    uniform mat4 projectionMatrix;
    uniform mat3 normalMatrix;
    uniform mat4 jointMatrices[MAX_JOINTS];

    layout (location = 0) in vec3 attrPosition;
    layout (location = 1) in vec3 attrNormal;
    layout (location = 3) in vec2 attrTexCoords;
    layout (location = 6) in vec4 attrJointIds;
    layout (location = 7) in vec4 attrJointWeights;

    out vec2 texCoords;
    out vec3 normal; // view space

    void main() {
        mat4 skinMatrix = attrJointWeights.x * jointMatrices[int(attrJointIds.x)]
                        + attrJointWeights.y * jointMatrices[int(attrJointIds.y)]
                        + attrJointWeights.z * jointMatrices[int(attrJointIds.z)]
                        + attrJointWeights.w * jointMatrices[int(attrJointIds.w)];

        texCoords = attrTexCoords;
        normal = normalMatrix * mat3(skinMatrix) * attrNormal;
        gl_Position = projectionMatrix * viewMatrix * modelMatrix * skinMatrix * vec4(attrPosition, 1.0);
    }
)"sv;

const auto frag = R"(
    #version 330 core

    uniform vec4 baseColorFactor;
    uniform sampler2D baseColorTexture;
    uniform vec3 lightDir; // view space

    in vec2 texCoords;
    in vec3 normal;

    out vec4 fragColor;

    void main() {
        vec4 base = baseColorFactor * texture2D(baseColorTexture, texCoords);
        float nDotL = max(dot(lightDir, normalize(normal)), 0.0);
        fragColor = vec4(base.rgb * nDotL, base.a);
        // fragColor = vec4(1.0);
        // fragColor = vec4(normal.rgb * 2.0 - 1.0, 1.0);
        // fragColor = base;
    }
)"sv;

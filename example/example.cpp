#include <iostream>

#include <glm/gtx/transform.hpp>
#include <glw.hpp>
#include <glwx.hpp>

#include "gltf.hpp"

#include "shaders.hpp"

using namespace std::literals;

struct Scene {
    using NodeIndex = size_t;
    using SkinIndex = size_t;
    using MaterialIndex = size_t;
    using PrimitiveIndex = size_t;
    using CameraIndex = size_t;

    struct Node {
        std::optional<NodeIndex> parent;
        std::vector<NodeIndex> children;
        glwx::Transform transform;
        std::vector<PrimitiveIndex> primitives;
        std::optional<SkinIndex> skin;
    };

    struct Skin {
        struct Joint {
            NodeIndex node;
            glm::mat4 inverseBindMatrix = glm::mat4(1.0f);
        };

        std::vector<glm::mat4> boneMatrices;
        std::vector<Joint> joints;
        NodeIndex rootNode;
    };

    struct Camera {
        NodeIndex node;
        glm::mat4 projection;
    };

    struct Material {
        glm::vec4 baseColorFactor { 1.0f, 1.0f, 1.0f, 1.0f };
        std::shared_ptr<glw::Texture> texture;
    };

    struct Primitive {
        glwx::Primitive drawable;
        std::optional<MaterialIndex> material;
    };

    struct Animation {
        enum class Interpolation { Step, Linear };
        enum class Destination { Translation, Scale, Rotation };

        template <typename T>
        struct ChannelBase {
            struct Keyframe {
                float time;
                T value;
            };

            Destination destination;
            NodeIndex nodeIndex;
            Interpolation interpolation;
            std::vector<Keyframe> keyframes {}; // sorted by time

            T interpolate(const T& from, const T& to, float alpha) const
            {
                assert(alpha >= 0.0f && alpha <= 1.0f);
                switch (interpolation) {
                case Interpolation::Step:
                    return from;
                case Interpolation::Linear:
                    if constexpr (std::is_same_v<T, glm::quat>) {
                        return glm::slerp(from, to, alpha);
                    } else if constexpr (std::is_same_v<T, glm::vec3>) {
                        return glm::mix(from, to, alpha);
                    }
                    assert(false && "Invalid type for animation channel");
                }
            }

            T getValue(float time) const
            {
                assert(!keyframes.empty());
                if (time <= keyframes.front().time)
                    return keyframes.front().value;
                if (time >= keyframes.back().time)
                    return keyframes.back().value;

                size_t toFrameIndex = 0;
                for (size_t i = 0; i < keyframes.size(); ++i) {
                    if (time <= keyframes[i].time) {
                        toFrameIndex = i;
                        break;
                    }
                }
                assert(toFrameIndex > 0);

                const auto& fromFrame = keyframes[toFrameIndex - 1];
                const auto& toFrame = keyframes[toFrameIndex];
                // time <= toFrame.time, time > fromFrame.time
                const auto alpha = (time - fromFrame.time) / (toFrame.time - fromFrame.time);

                return interpolate(fromFrame.value, toFrame.value, alpha);
            }

            void apply(std::vector<Node>& nodes, float time) const
            {
                auto& node = nodes[nodeIndex];
                if constexpr (std::is_same_v<T, glm::vec3>) {
                    switch (destination) {
                    case Destination::Translation:
                        node.transform.setPosition(getValue(time));
                        break;
                    case Destination::Scale:
                        node.transform.setScale(getValue(time));
                        break;
                    default:
                        assert(false && "Invalid destination for type");
                    }
                } else if constexpr (std::is_same_v<T, glm::quat>) {
                    assert(destination == Destination::Rotation);
                    node.transform.setOrientation(getValue(time));
                }
            }
        };

        // I don't think I like this
        using Channel = std::variant<ChannelBase<glm::vec3>, ChannelBase<glm::quat>>;
        std::vector<Channel> channels;

        float getDuration() const
        {
            float dur = 0.0f;
            for (const auto& ch : channels) {
                const auto channelDur
                    = std::visit([](auto&& c) { return c.keyframes.back().time; }, ch);
                if (channelDur > dur)
                    dur = channelDur;
            }
            return dur;
        }

        void apply(std::vector<Node>& nodes, float time) const
        {
            const auto wrappedTime = std::fmod(time, getDuration());
            for (const auto& ch : channels) {
                std::visit([&nodes, wrappedTime](auto&& c) { c.apply(nodes, wrappedTime); }, ch);
            }
        }
    };

    glw::ShaderProgram normalShader;
    glw::ShaderProgram skinningShader;
    std::vector<glw::Buffer> buffers;
    std::vector<Primitive> primitives;
    std::vector<Node> nodes;
    std::vector<Camera> cameras;
    std::vector<Material> materials;
    std::vector<Skin> skins;
    Material defaultMaterial;
    std::vector<NodeIndex> rootNodes;
    glwx::Aabb bbox;
    std::vector<Animation> animations;

    glm::mat4 getFullTransform(NodeIndex nodeIndex) const
    {
        const auto& node = nodes[nodeIndex];
        if (node.parent)
            return getFullTransform(*node.parent) * node.transform.getMatrix();
        return node.transform.getMatrix();
    }

    void updateBoneMatrices(SkinIndex skinIndex)
    {
        auto& skin = skins[skinIndex];
        const auto rootInverse = glm::inverse(getFullTransform(skin.rootNode));
        for (size_t i = 0; i < skin.joints.size(); ++i) {
            skin.boneMatrices[i] = rootInverse * getFullTransform(skin.joints[i].node)
                * skin.joints[i].inverseBindMatrix;
        }
    }

    void updateSkins()
    {
        for (size_t i = 0; i < skins.size(); ++i)
            updateBoneMatrices(i);
    }

    void drawPrimitive(PrimitiveIndex primitiveIndex, const glw::ShaderProgram& shader) const
    {
        const auto& primitive = primitives[primitiveIndex];

        const auto& material
            = primitive.material ? materials[*primitive.material] : defaultMaterial;
        shader.setUniform("baseColorFactor", material.baseColorFactor);
        material.texture->bind(0);
        shader.setUniform("baseColorTexture", 0);

        primitive.drawable.draw();
    }

    void drawNode(NodeIndex nodeIndex, const glm::mat4& parentModelMatrix,
        const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix) const
    {
        const auto& node = nodes[nodeIndex];

        const auto modelMatrix = parentModelMatrix * node.transform.getMatrix();
        if (!node.primitives.empty()) {
            const auto& shader = node.skin ? skinningShader : normalShader;
            shader.bind();
            shader.setUniform("lightDir", glm::vec3(0.0f, 0.0f, 1.0f));
            shader.setUniform("modelMatrix", modelMatrix);
            shader.setUniform("viewMatrix", viewMatrix);
            shader.setUniform("projectionMatrix", projectionMatrix);
            const auto modelViewMatrix = viewMatrix * modelMatrix;
            const auto normalMatrix = glm::mat3(glm::transpose(glm::inverse(modelViewMatrix)));
            shader.setUniform("normalMatrix", normalMatrix);

            if (node.skin) {
                const auto& matrices = skins[*node.skin].boneMatrices;
                assert(matrices.size() <= 32);
                shader.setUniform("jointMatrices", matrices.data(), matrices.size());
            }

            for (const auto primitive : node.primitives)
                drawPrimitive(primitive, shader);
        } else if (node.skin) {
            LOG_WARNING("skin but not primitives!");
        }

        for (const auto child : node.children)
            drawNode(child, modelMatrix, viewMatrix, projectionMatrix);
    }

    void draw(CameraIndex cameraIndex) const
    {
        assert(cameraIndex < cameras.size());
        const auto& camera = cameras[cameraIndex];
        const auto view = glm::inverse(getFullTransform(camera.node));

        for (const auto node : rootNodes) {
            drawNode(node, glm::mat4(1.0f), view, camera.projection);
        }
    }
};

size_t getAttributeLocation(std::string_view id)
{
    if (id == "POSITION")
        return AttributeLocations::Position;
    else if (id == "NORMAL")
        return AttributeLocations::Normal;
    else if (id == "TANGENT")
        return AttributeLocations::Tangent;
    else if (id == "TEXCOORD_0")
        return AttributeLocations::TexCoord0;
    else if (id == "TEXCOORD_1")
        return AttributeLocations::TexCoord1;
    else if (id == "COLOR_0")
        return AttributeLocations::Color0;
    else if (id == "JOINTS_0")
        return AttributeLocations::Joints0;
    else if (id == "WEIGHTS_0")
        return AttributeLocations::Weights0;
    else
        assert(false && "Invalid attribute id");
}

template <typename Container>
glm::vec3 makeVec3(const Container& vals)
{
    assert(vals.size() == 3);
    return glm::vec3(
        static_cast<float>(vals[0]), static_cast<float>(vals[1]), static_cast<float>(vals[2]));
}

template <typename Container>
glm::vec4 makeVec4(const Container& vals)
{
    assert(vals.size() == 4);
    return glm::vec4(static_cast<float>(vals[0]), static_cast<float>(vals[1]),
        static_cast<float>(vals[2]), static_cast<float>(vals[3]));
}

template <typename Container>
glm::mat4 makeMat4(const Container& vals)
{
    assert(vals.size() == 16);
    glm::mat4 ret;
    const auto ptr = glm::value_ptr(ret);
    for (size_t i = 0; i < 16; ++i)
        ptr[i] = vals[i];
    return ret;
}

Scene::Animation::Interpolation convertInterpolation(gltf::Animation::Sampler::Interpolation interp)
{
    switch (interp) {
    case gltf::Animation::Sampler::Interpolation::Linear:
        return Scene::Animation::Interpolation::Linear;
    case gltf::Animation::Sampler::Interpolation::Step:
        return Scene::Animation::Interpolation::Step;
    default:
        assert(false && "Invalid enum value");
        return Scene::Animation::Interpolation {};
    }
}

template <typename T>
void readKeyframeTimes(
    std::vector<T>& keyframes, const gltf::Accessor& accessor, const uint8_t* data)
{
    assert(accessor.type == gltf::Accessor::Type::Scalar);
    assert(accessor.componentType == gltf::Accessor::ComponentType::Float);
    const auto fdata = reinterpret_cast<const float*>(data);
    assert(keyframes.size() >= accessor.count);
    for (size_t i = 0; i < accessor.count; ++i)
        keyframes[i].time = fdata[i];
}

void readKeyframeValues(std::vector<Scene::Animation::ChannelBase<glm::vec3>::Keyframe>& keyframes,
    const gltf::Accessor& accessor, const uint8_t* data)
{
    assert(accessor.type == gltf::Accessor::Type::Vec3);
    assert(accessor.componentType == gltf::Accessor::ComponentType::Float);
    assert(keyframes.size() >= accessor.count);
    const auto fdata = reinterpret_cast<const float*>(data);
    for (size_t i = 0; i < accessor.count; ++i)
        keyframes[i].value = glm::make_vec3(fdata + i * 3);
}

void readKeyframeValues(std::vector<Scene::Animation::ChannelBase<glm::quat>::Keyframe>& keyframes,
    const gltf::Accessor& accessor, const uint8_t* data)
{
    assert(accessor.type == gltf::Accessor::Type::Vec4);
    assert(accessor.componentType == gltf::Accessor::ComponentType::Float
        && "Normalized ints unimplemented");
    assert(keyframes.size() >= accessor.count);
    const auto fdata = reinterpret_cast<const float*>(data);
    for (size_t i = 0; i < accessor.count; ++i)
        keyframes[i].value = glm::make_quat(fdata + i * 4);
}

template <typename T>
std::vector<uint8_t> flipIndexData(const T* data, size_t count)
{
    std::vector<uint8_t> ret(sizeof(T) * count);
    auto outData = reinterpret_cast<T*>(ret.data());
    for (size_t i = 0; i < count; i += 3) {
        outData[i + 0] = data[i + 2];
        outData[i + 1] = data[i + 1];
        outData[i + 2] = data[i + 0];
    }
    return ret;
}

std::vector<uint8_t> flipIndexData(
    const uint8_t* data, gltf::Accessor::ComponentType type, size_t count)
{
    assert(count % 3 == 0);
    switch (type) {
    case gltf::Accessor::ComponentType::UnsignedByte:
        return flipIndexData(reinterpret_cast<const uint8_t*>(data), count);
    case gltf::Accessor::ComponentType::UnsignedShort:
        return flipIndexData(reinterpret_cast<const uint16_t*>(data), count);
    case gltf::Accessor::ComponentType::UnsignedInt:
        return flipIndexData(reinterpret_cast<const uint32_t*>(data), count);
    default:
        assert(false && "Invalid index type");
    }
}

// This aspectRatio parameter is so fucking weird
std::optional<Scene> loadGltf(const std::filesystem::path& path, float aspectRatio)
{
    const auto gltfFileOpt = gltf::load(path);
    if (!gltfFileOpt) {
        std::cerr << "Could not load GLTF file" << std::endl;
        return std::nullopt;
    }
    const auto& gltfFile = *gltfFileOpt;
    // can't do 0 scenes, won't do > 1 scene
    // I can' just ignore the other scenes, because I would have to ignore nodes and stuff too
    // but then the indices would not be the same and this loading code would be way more
    // complex
    assert(gltfFile.scenes.size() == 1);

    Scene scene;

    scene.normalShader = glwx::makeShaderProgram(vert, frag).value();
    scene.skinningShader = glwx::makeShaderProgram(skinningVert, frag).value();

    // I don't want to figure out which buffers, meshes or whatever I need, I'll just take them
    // all But I still need an index map for bufferViews, because I can't turn them all into
    // glwx::Buffers
    std::unordered_map<gltf::BufferViewIndex, size_t> bufferViewIndexMap;
    for (size_t i = 0; i < gltfFile.bufferViews.size(); ++i) {
        const auto& gbufferView = gltfFile.bufferViews[i];
        if (gbufferView.target) {
            const auto target = static_cast<glw::Buffer::Target>(*gbufferView.target);
            const auto data = gltfFile.getBufferViewData(i);
            glw::Buffer buffer;
            // LOG_DEBUG("BufferView {}: {}", scene.buffers.size(), glwx::toHexStream(dataPtr,
            // gbufferView.byteLength));
            buffer.data(target, glw::Buffer::UsageHint::StaticDraw, data.first, data.second);
            scene.buffers.push_back(std::move(buffer));
            bufferViewIndexMap.emplace(i, scene.buffers.size() - 1);
        }
    }

    auto defaultTexture = std::make_shared<glw::Texture>(glwx::makeTexture2D(glm::vec4(1.0f)));
    scene.defaultMaterial.texture = defaultTexture;

    std::vector<std::shared_ptr<glw::Texture>> textures;
    for (size_t i = 0; i < gltfFile.textures.size(); ++i) {
        const auto& gtexture = gltfFile.textures[i];
        assert(gtexture.source);
        const auto data = gltfFile.getImageData(*gtexture.source);
        auto minFilter = glw::Texture::MinFilter::LinearMipmapNearest;
        auto magFilter = glw::Texture::MagFilter::Linear;
        auto wrapS = glw::Texture::WrapMode::Repeat;
        auto wrapT = glw::Texture::WrapMode::Repeat;
        if (gtexture.sampler) {
            const auto& sampler = gltfFile.samplers[*gtexture.sampler];
            if (sampler.minFilter)
                minFilter = static_cast<glw::Texture::MinFilter>(*sampler.minFilter);
            if (sampler.magFilter)
                magFilter = static_cast<glw::Texture::MagFilter>(*sampler.magFilter);
            wrapS = static_cast<glw::Texture::WrapMode>(sampler.wrapS);
            wrapT = static_cast<glw::Texture::WrapMode>(sampler.wrapT);
        }
        const auto mipmaps = static_cast<GLenum>(minFilter)
            >= static_cast<GLenum>(glw::Texture::MinFilter::NearestMipmapNearest);
        auto tex = glwx::makeTexture2D(data.first, data.second, mipmaps);
        if (!tex) {
            std::cerr << "Could not load texture" << std::endl;
            return std::nullopt;
        }
        tex->setMinFilter(minFilter);
        tex->setMagFilter(magFilter);
        tex->setWrap(wrapS, wrapT);
        textures.push_back(std::make_shared<glw::Texture>(std::move(*tex)));
    }

    for (size_t i = 0; i < gltfFile.materials.size(); ++i) {
        const auto& gmaterial = gltfFile.materials[i];
        auto& material = scene.materials.emplace_back();
        material.texture = defaultTexture;
        if (gmaterial.pbrMetallicRoughness) {
            const auto& pbr = *gmaterial.pbrMetallicRoughness;
            material.baseColorFactor = makeVec4(pbr.baseColorFactor);
            if (pbr.baseColorTexture) {
                const auto& texInfo = *pbr.baseColorTexture;
                assert(texInfo.texCoord == 0);
                material.texture = textures[texInfo.index];
            }
        }
    }

    // A single mesh.primitive corresponds to multiple primitives, but after this block of code
    // we will never consider a mesh again (just primitives)
    std::vector<std::vector<size_t>> meshPrimitivesMap;
    std::vector<glwx::Aabb> meshBboxs;
    for (size_t i = 0; i < gltfFile.meshes.size(); ++i) {
        const auto& gmesh = gltfFile.meshes[i];
        auto& primitives = meshPrimitivesMap.emplace_back();
        auto& bbox = meshBboxs.emplace_back();
        for (size_t p = 0; p < gmesh.primitives.size(); ++p) {
            const auto& gprimitive = gmesh.primitives[p];
            glwx::Primitive primitive(static_cast<glw::DrawMode>(gprimitive.mode));

            // This is kinda dumb, but I don't want to think
            std::set<gltf::BufferViewIndex> processedBufferViews;
            bool hasNormals = false;
            for (const auto& attribute : gprimitive.attributes) {
                assert(gltfFile.accessors[attribute.accessor].bufferView);
                processedBufferViews.insert(*gltfFile.accessors[attribute.accessor].bufferView);
                hasNormals = hasNormals || attribute.id == "NORMAL";
            }
            if (!hasNormals) {
                LOG_ERROR("Primitive {} of Mesh {} does not have normals!", p, i);
            }

            for (const auto view : processedBufferViews) {
                auto& bufferView = gltfFile.bufferViews[view];
                glw::VertexFormat vfmt;
                std::optional<size_t> vertexCount;
                for (const auto& attribute : gprimitive.attributes) {
                    auto& accessor = gltfFile.accessors[attribute.accessor];
                    if (view == accessor.bufferView.value()) {
                        if (attribute.id == "POSITION") {
                            bbox.fit(makeVec3(accessor.min));
                            bbox.fit(makeVec3(accessor.max));
                        }

                        const auto count = static_cast<size_t>(accessor.type);
                        assert(count >= 1 && count <= 4);
                        const auto componentType
                            = static_cast<glw::AttributeType>(accessor.componentType);
                        vfmt.add(accessor.byteOffset, getAttributeLocation(attribute.id), count,
                            componentType, accessor.normalized);
                        vertexCount
                            = vertexCount ? std::min(*vertexCount, accessor.count) : accessor.count;
                    }
                }

                if (bufferView.byteStride)
                    vfmt.setStride(*bufferView.byteStride);
                primitive.addVertexBuffer(scene.buffers[bufferViewIndexMap.at(view)], vfmt);
                primitive.vertexRange = glwx::Primitive::Range { 0, vertexCount.value() };
            }

            if (gprimitive.indices) {
                auto& accessor = gltfFile.accessors[*gprimitive.indices];
                const auto type = static_cast<glw::IndexType>(accessor.componentType);
                assert(type == glw::IndexType::U8 || type == glw::IndexType::U16
                    || type == glw::IndexType::U32);
                primitive.setIndexBuffer(
                    scene.buffers[bufferViewIndexMap.at(accessor.bufferView.value())], type);
                primitive.indexRange
                    = glwx::Primitive::Range { accessor.byteOffset / glw::getIndexTypeSize(type),
                          accessor.count };
            }

            scene.primitives.push_back(
                Scene::Primitive { std::move(primitive), gprimitive.material });
            primitives.push_back(scene.primitives.size() - 1);
        }
    }

    for (const auto& gskin : gltfFile.skins) {
        auto& skin = scene.skins.emplace_back();

        for (const auto node : gskin.joints)
            skin.joints.push_back(Scene::Skin::Joint { node });

        skin.boneMatrices.resize(skin.joints.size(), glm::mat4(1.0f));

        bool found = false;
        for (size_t n = 0; n < gltfFile.nodes.size(); ++n) {
            const auto& node = gltfFile.nodes[n];
            if (node.skin && *node.skin == scene.skins.size() - 1) {
                assert(!found && "Skin used for multiple nodes");
                skin.rootNode = n;
                found = true;
            }
        }
        assert(found && "Could not determine root node of skin");

        if (gskin.inverseBindMatrices) {
            auto& acc = gltfFile.accessors[*gskin.inverseBindMatrices];
            assert(acc.componentType == gltf::Accessor::ComponentType::Float);
            assert(acc.type == gltf::Accessor::Type::Mat4);
            assert(acc.count == skin.joints.size());
            const auto data = gltfFile.getAccessorData(*gskin.inverseBindMatrices);
            assert(data.second <= 16 * sizeof(float) * acc.count);
            const auto matData = reinterpret_cast<const glm::mat4*>(data.first);
            for (size_t i = 0; i < acc.count; ++i)
                skin.joints[i].inverseBindMatrix = matData[i];
        }
    }

    std::set<size_t> flipMeshes;
    for (const auto& gnode : gltfFile.nodes) {
        auto& node = scene.nodes.emplace_back();
        // Matrix has to be TRS decomposable
        const auto trafo = makeMat4(gnode.getTransformMatrix());
        node.transform.setMatrix(trafo);
        const auto fullTrafo = gnode.parent
            ? makeMat4(gltfFile.nodes[*gnode.parent].getTransformMatrix()) * trafo
            : trafo;
        if (gnode.mesh) {
            node.primitives = meshPrimitivesMap[*gnode.mesh];

            if (glm::determinant(glm::mat3(fullTrafo)) < 0.0f) {
                flipMeshes.insert(*gnode.mesh);
            }

            assert(meshBboxs[*gnode.mesh].valid());
            scene.bbox.fit(meshBboxs[*gnode.mesh].transformed(node.transform.getMatrix()));
        }
        node.children = gnode.children;
        node.parent = gnode.parent;

        if (gnode.camera) {
            const auto& gcamera = gltfFile.cameras[*gnode.camera];
            auto& camera = scene.cameras.emplace_back();
            camera.node = scene.nodes.size() - 1;
            camera.projection = makeMat4(gcamera.getProjection(aspectRatio));
        }

        if (gnode.skin) {
            node.skin = *gnode.skin;
        }
    }

    // If some nodes need a mesh (-> buffer) flipped and others don't this will go wrong
    for (const auto mesh : flipMeshes) {
        LOG_DEBUG("Flip mesh {}", mesh);
        for (const auto& prim : gltfFile.meshes[mesh].primitives) {
            assert(prim.indices && "Flipping of non-indexed meshes is not unimplemented");
            assert(prim.mode == gltf::Mesh::Primitive::Mode::Triangles
                && "Flipping of non-triangle meshes is not implemented");
            const auto& acc = gltfFile.accessors[*prim.indices];
            const auto flippedData = flipIndexData(
                gltfFile.getAccessorData(*prim.indices).first, acc.componentType, acc.count);
            const auto& buf = scene.buffers[bufferViewIndexMap[*acc.bufferView]];
            buf.subData(glw::Buffer::Target::ElementArray, acc.byteOffset, flippedData);
        }
    }

    auto& gscene = gltfFile.scenes[0];
    for (const auto node : gscene.nodes)
        scene.rootNodes.push_back(node);

    for (const auto& ganim : gltfFile.animations) {
        auto& anim = scene.animations.emplace_back();
        for (const auto& gchannel : ganim.channels) {
            if (!gchannel.target.node)
                continue;
            const auto& gsampler = ganim.samplers[gchannel.sampler];

            auto& channel = anim.channels.emplace_back();
            switch (gchannel.target.path) {
            case gltf::Animation::Channel::Target::Path::Translation:
                channel = Scene::Animation::ChannelBase<glm::vec3> {
                    Scene::Animation::Destination::Translation, *gchannel.target.node,
                    convertInterpolation(gsampler.interpolation)
                };
                break;
            case gltf::Animation::Channel::Target::Path::Rotation:
                channel = Scene::Animation::ChannelBase<glm::quat> {
                    Scene::Animation::Destination::Rotation, *gchannel.target.node,
                    convertInterpolation(gsampler.interpolation)
                };
                break;
            case gltf::Animation::Channel::Target::Path::Scale:
                channel = Scene::Animation::ChannelBase<glm::vec3> {
                    Scene::Animation::Destination::Scale, *gchannel.target.node,
                    convertInterpolation(gsampler.interpolation)
                };
                break;
            case gltf::Animation::Channel::Target::Path::Weights:
                assert(false && "Morph targets unimplemented");
            }

            const auto& time = gltfFile.accessors[gsampler.input];
            const auto timeData = gltfFile.getAccessorData(gsampler.input).first;
            const auto& value = gltfFile.accessors[gsampler.output];
            const auto valueData = gltfFile.getAccessorData(gsampler.output).first;
            std::visit(
                [&time, timeData, &value, valueData](auto&& ch) {
                    assert(time.count == value.count);
                    ch.keyframes.resize(time.count);
                    readKeyframeTimes(ch.keyframes, time, timeData);
                    readKeyframeValues(ch.keyframes, value, valueData);
                },
                channel);
        }
    }

    // I don't want to render them, so I'll just print them
    for (size_t i = 0; i < gltfFile.lights.size(); ++i) {
        const auto& light = gltfFile.lights[i];
        LOG_DEBUG("light {} ({})", i, light.name);
        LOG_DEBUG("color: {}", makeVec3(light.color));
        LOG_DEBUG("intensity: {}", light.intensity);
        if (const auto directional = std::get_if<gltf::Light::Directional>(&light.parameters)) {
            LOG_DEBUG("type: directional");
        } else if (const auto point = std::get_if<gltf::Light::Point>(&light.parameters)) {
            LOG_DEBUG("type: point");
            LOG_DEBUG("range: {}", point->range);
        } else if (const auto spot = std::get_if<gltf::Light::Spot>(&light.parameters)) {
            LOG_DEBUG("type: spot");
            LOG_DEBUG("range: {}", spot->range);
            LOG_DEBUG("innerConeAngle: {}", spot->innerConeAngle);
            LOG_DEBUG("outerConeAngle: {}", spot->outerConeAngle);
        } else {
            assert(false && "Invalid light type");
        }

        for (size_t n = 0; n < gltfFile.nodes.size(); ++n) {
            const auto& nlight = gltfFile.nodes[n].light;
            if (nlight && *nlight == i)
                LOG_DEBUG("used by node: {}", n);
        }
    }

    return scene;
}

int main(int argc, char** argv)
{
    const std::vector<std::string_view> args(argv + 1, argv + argc);
    if (args.empty()) {
        LOG_CRITICAL("Usage: example-gltf <file>");
        return EXIT_FAILURE;
    }

    glwx::Window::Properties props;
    props.msaaSamples = 8;
    const auto window = glwx::makeWindow("GLTF Example", 1024, 768, props).value();
    glw::State::instance().setViewport(window.getSize().x, window.getSize().y);

#ifndef NDEBUG
    glwx::debug::init();
#endif

    const auto aspect = static_cast<float>(window.getSize().x) / window.getSize().y;
    auto scene = loadGltf(args[0], aspect);
    if (!scene) {
        LOG_CRITICAL("Could not load glTF file");
        return EXIT_FAILURE;
    }
    assert(scene->bbox.valid());
    const auto center = scene->bbox.center();
    const auto size = args.size() > 1 ? std::stof(std::string(args[1]))
                                      : glm::length(scene->bbox.size()) * 0.5f;
    LOG_DEBUG("size: {}", size);

    auto& node = scene->nodes.emplace_back();
    node.transform = glm::translate(glm::mat4(1.0f), center + glm::vec3(0.0f, 0.0f, size * 2.0f));
    auto& camera = *scene->cameras.emplace(scene->cameras.begin());
    camera.node = scene->nodes.size() - 1;
    camera.projection = glm::perspective(glm::radians(45.0f), aspect, size * 0.01f, size * 10.0f);

    size_t cameraIndex = 0;
    size_t animIndex = 0;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    bool wireframe = false;

    SDL_Event event;
    bool running = true;
    float time = glwx::getTime();
    while (running) {
        while (SDL_PollEvent(&event) != 0) {
            switch (event.type) {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                case SDLK_LEFT:
                    cameraIndex = (cameraIndex - 1) % scene->cameras.size();
                    LOG_DEBUG("current camera: {}", cameraIndex);
                    break;
                case SDLK_RIGHT:
                    cameraIndex = (cameraIndex + 1) % scene->cameras.size();
                    LOG_DEBUG("current camera: {}", cameraIndex);
                    break;
                case SDLK_h:
                    wireframe = !wireframe;
                    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
                    break;
                case SDLK_n:
                    animIndex = (animIndex + 1) % scene->animations.size();
                    LOG_DEBUG("current animation: {}", animIndex);
                    break;
                case SDLK_m:
                    animIndex = (animIndex - 1) % scene->animations.size();
                    LOG_DEBUG("current animation: {}", animIndex);
                    break;
                case SDLK_ESCAPE:
                    running = false;
                    break;
                }
                break;
            }
        }

        const auto now = glwx::getTime();
        const auto dt = now - time;
        time = now;

        int mouseDx = 0, mouseDy = 0;
        const auto mouseButton = SDL_GetRelativeMouseState(&mouseDx, &mouseDy);
        glm::vec2 look(0.0f);
        if (mouseButton & SDL_BUTTON(1)) {
            const auto sensitivity = 0.01f;
            look = glm::vec2(mouseDx * sensitivity, mouseDy * sensitivity);
        }
        const auto kbState = SDL_GetKeyboardState(nullptr);
        auto key = [&kbState](int scancode) { return kbState[scancode] ? 1.0f : 0.0f; };
        const auto forward = key(SDL_SCANCODE_S) - key(SDL_SCANCODE_W);
        const auto sideways = key(SDL_SCANCODE_D) - key(SDL_SCANCODE_A);
        const auto updown = key(SDL_SCANCODE_R) - key(SDL_SCANCODE_F);
        const auto speed = (kbState[SDL_SCANCODE_LSHIFT] ? 2.f : 0.4f) * size;
        const auto move = speed * dt * glm::vec3(sideways, updown, forward);
        if (glm::length(look) > 0.0f || glm::length(move) > 0.0f) {
            auto& node = scene->nodes[scene->cameras[cameraIndex].node];
            node.transform.rotate(glm::angleAxis(-look.x, glm::vec3(0.0f, 1.0f, 0.0f)));
            node.transform.rotateLocal(glm::angleAxis(-look.y, glm::vec3(1.0f, 0.0f, 0.0f)));
            node.transform.moveLocal(move);
        }

        if (!scene->animations.empty())
            scene->animations[animIndex].apply(scene->nodes, now);
        scene->updateSkins();

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        scene->draw(cameraIndex);

        window.swap();
    }

    return 0;
}

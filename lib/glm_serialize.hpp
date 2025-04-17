#ifndef GLM_SERIALIZE_HPP
#define GLM_SERIALIZE_HPP

#include <cereal/cereal.hpp>
#include <glm/glm.hpp>

namespace glm {

template <class Archive, typename T, glm::precision P>
void save(Archive& ar, const glm::tvec2<T, P>& vec) {
    ar(cereal::make_nvp("x", vec.x), cereal::make_nvp("y", vec.y));
}

template <class Archive, typename T, glm::precision P>
void load(Archive& ar, glm::tvec2<T, P>& vec) {
    ar(cereal::make_nvp("x", vec.x), cereal::make_nvp("y", vec.y));
}

template <class Archive, typename T, glm::precision P>
void save(Archive& ar, const glm::tvec3<T, P>& vec) {
    ar(cereal::make_nvp("x", vec.x), cereal::make_nvp("y", vec.y), cereal::make_nvp("z", vec.z));
}

template <class Archive, typename T, glm::precision P>
void load(Archive& ar, glm::tvec3<T, P>& vec) {
    ar(cereal::make_nvp("x", vec.x), cereal::make_nvp("y", vec.y), cereal::make_nvp("z", vec.z));
}

template <class Archive, typename T, glm::precision P>
void save(Archive& ar, const glm::tvec4<T, P>& vec) {
    ar(cereal::make_nvp("x", vec.x), cereal::make_nvp("y", vec.y), cereal::make_nvp("z", vec.z), cereal::make_nvp("w", vec.w));
}

template <class Archive, typename T, glm::precision P>
void load(Archive& ar, glm::tvec4<T, P>& vec) {
    ar(cereal::make_nvp("x", vec.x), cereal::make_nvp("y", vec.y), cereal::make_nvp("z", vec.z), cereal::make_nvp("w", vec.w));
}

template <class Archive, typename T, glm::precision P>
void save(Archive& ar, const glm::tmat3x3<T, P>& mat) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ar(cereal::make_nvp("m" + std::to_string(i) + std::to_string(j), mat[i][j]));
        }
    }
}

template <class Archive, typename T, glm::precision P>
void load(Archive& ar, glm::tmat3x3<T, P>& mat) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ar(cereal::make_nvp("m" + std::to_string(i) + std::to_string(j), mat[i][j]));
        }
    }
}

template <class Archive, typename T, glm::precision P>
void save(Archive& ar, const glm::tmat4x4<T, P>& mat) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            ar(cereal::make_nvp("m" + std::to_string(i) + std::to_string(j), mat[i][j]));
        }
    }
}

template <class Archive, typename T, glm::precision P>
void load(Archive& ar, glm::tmat4x4<T, P>& mat) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            ar(cereal::make_nvp("m" + std::to_string(i) + std::to_string(j), mat[i][j]));
        }
    }
}

template <class Archive, typename T, glm::precision P>
void save(Archive& ar, const glm::tquat<T, P>& quat) {
    ar(cereal::make_nvp("x", quat.x), cereal::make_nvp("y", quat.y), cereal::make_nvp("z", quat.z), cereal::make_nvp("w", quat.w));
}

template <class Archive, typename T, glm::precision P>
void load(Archive& ar, glm::tquat<T, P>& quat) {
    ar(cereal::make_nvp("x", quat.x), cereal::make_nvp("y", quat.y), cereal::make_nvp("z", quat.z), cereal::make_nvp("w", quat.w));
}

} // namespace glm

#endif // GLM_SERIALIZE_HPP
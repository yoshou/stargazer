set(COALSACK_BUILD_DIR "${CMAKE_SOURCE_DIR}/coalsack/build")
set(GTEST_INCLUDE_DIR "${COALSACK_BUILD_DIR}/third-party/googletest/googletest/include")
set(GTEST_LIB_DIR "${COALSACK_BUILD_DIR}/third-party/googletest-build/lib")

add_executable(test_config
    ${CMAKE_CURRENT_LIST_DIR}/test_config.cpp
    ${CMAKE_SOURCE_DIR}/lib/config.cpp
)

target_include_directories(test_config SYSTEM PRIVATE
    ${GTEST_INCLUDE_DIR}
)

target_include_directories(test_config PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/lib
)

target_link_libraries(test_config PRIVATE
    ${GTEST_LIB_DIR}/libgtest.a
    ${GTEST_LIB_DIR}/libgtest_main.a
    nlohmann_json::nlohmann_json
    pthread
)

target_compile_features(test_config PRIVATE cxx_std_17)

target_compile_definitions(test_config PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/lib/testdata"
)

# ---------------------------------------------------------------------------
# test_dust3r_inference: Phase-1 inference verification
# ---------------------------------------------------------------------------
add_executable(test_dust3r_inference
    ${CMAKE_CURRENT_LIST_DIR}/test_dust3r_inference.cpp
)

target_include_directories(test_dust3r_inference SYSTEM PRIVATE
    ${GTEST_INCLUDE_DIR}
)

target_include_directories(test_dust3r_inference PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/lib
)

target_link_libraries(test_dust3r_inference PRIVATE
    ${GTEST_LIB_DIR}/libgtest.a
    ${GTEST_LIB_DIR}/libgtest_main.a
    stargazer_dust3r
    Ceres::ceres
    opencv_imgproc
    opencv_calib3d
    pthread
)

target_compile_features(test_dust3r_inference PRIVATE cxx_std_20)
set_target_properties(test_dust3r_inference PROPERTIES CXX_STANDARD 20)
add_test(NAME test_dust3r_inference COMMAND test_dust3r_inference)

# ---------------------------------------------------------------------------
# test_dust3r_alignment: Phase-2 alignment verification
# ---------------------------------------------------------------------------
add_executable(test_dust3r_alignment
    ${CMAKE_CURRENT_LIST_DIR}/test_dust3r_alignment.cpp
)

target_include_directories(test_dust3r_alignment SYSTEM PRIVATE
    ${GTEST_INCLUDE_DIR}
)

target_include_directories(test_dust3r_alignment PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/lib
)

target_link_libraries(test_dust3r_alignment PRIVATE
    ${GTEST_LIB_DIR}/libgtest.a
    ${GTEST_LIB_DIR}/libgtest_main.a
    stargazer_dust3r
    Ceres::ceres
    pthread
)

target_compile_features(test_dust3r_alignment PRIVATE cxx_std_20)
set_target_properties(test_dust3r_alignment PROPERTIES CXX_STANDARD 20)
add_test(NAME test_dust3r_alignment COMMAND test_dust3r_alignment)

# ---------------------------------------------------------------------------
# test_dust3r_calibration: Phase 3-5 unit tests (coord transform, merge, synthetic BA)
# ---------------------------------------------------------------------------
add_executable(test_dust3r_calibration
    ${CMAKE_CURRENT_LIST_DIR}/test_dust3r_calibration.cpp
)

target_include_directories(test_dust3r_calibration SYSTEM PRIVATE
    ${GTEST_INCLUDE_DIR}
)

target_include_directories(test_dust3r_calibration PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/lib
)

target_link_libraries(test_dust3r_calibration PRIVATE
    ${GTEST_LIB_DIR}/libgtest.a
    ${GTEST_LIB_DIR}/libgtest_main.a
    stargazer_dust3r
    Ceres::ceres
    opencv_imgproc
    opencv_calib3d
    opencv_flann
    pthread
)

target_compile_features(test_dust3r_calibration PRIVATE cxx_std_20)
set_target_properties(test_dust3r_calibration PROPERTIES CXX_STANDARD 20)
add_test(NAME test_dust3r_calibration COMMAND test_dust3r_calibration)

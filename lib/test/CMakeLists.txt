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

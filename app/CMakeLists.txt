cmake_minimum_required(VERSION 3.18)

# Find glslc compiler
find_program(GLSLC glslc REQUIRED)

# Shader compilation function
function(compile_shader SHADER_SOURCE SHADER_OUTPUT)
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    get_filename_component(SHADER_DIR ${SHADER_SOURCE} DIRECTORY)
    set(SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)
    set(SHADER_INC ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv.inc)
    
    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${GLSLC} ${SHADER_SOURCE} -o ${SHADER_SPV}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling shader ${SHADER_NAME}"
        VERBATIM
    )
    
    add_custom_command(
        OUTPUT ${SHADER_INC}
        COMMAND python3 ${SHADER_DIR}/convert_shader.py ${SHADER_SPV} ${SHADER_INC}
        DEPENDS ${SHADER_SPV} ${SHADER_DIR}/convert_shader.py
        COMMENT "Converting ${SHADER_NAME}.spv to C header"
        VERBATIM
    )
    
    set(${SHADER_OUTPUT} ${SHADER_INC} PARENT_SCOPE)
endfunction()

# Compile shaders
compile_shader(${CMAKE_CURRENT_LIST_DIR}/shaders/simple3d.vert VERT_SHADER_INC)
compile_shader(${CMAKE_CURRENT_LIST_DIR}/shaders/simple3d.frag FRAG_SHADER_INC)

# Create custom target for shader compilation
add_custom_target(stargazer_shaders ALL
    DEPENDS ${VERT_SHADER_INC} ${FRAG_SHADER_INC}
)

add_executable(stargazer_viewer
    ${CMAKE_CURRENT_LIST_DIR}/viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/views.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gui.cpp
    ${CMAKE_CURRENT_LIST_DIR}/render3d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/capture_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calibration_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/viewer_app.cpp
)

add_dependencies(stargazer_viewer
    stargazer
    stargazer_shaders
    stargazer_sensor_grpc_proto
    stargazer_voxelpose
    stargazer_preprocess
    stargazer_mvpose
    stargazer_mvp
)

target_include_directories(stargazer_viewer
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
)

target_compile_definitions(stargazer_viewer
    PUBLIC
    $<$<BOOL:${MSVC}>:_USE_MATH_DEFINES>
    $<$<BOOL:${MSVC}>:GLOG_NO_ABBREVIATED_SEVERITIES>
)

target_link_directories(stargazer_viewer
    PRIVATE
    ${VCPKG_ROOT}/installed/x64-linux/lib/
)

target_link_libraries(stargazer_viewer
    PRIVATE
    stargazer
    stargazer_sensor_grpc_proto
    stargazer_voxelpose
    stargazer_preprocess
    stargazer_mvpose
    stargazer_mvp
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_features2d
    opencv_calib3d
    opencv_aruco
    $<$<NOT:$<BOOL:${WIN32}>>:vulkan>
    $<$<NOT:$<BOOL:${WIN32}>>:dl>
    glfw
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
    spdlog::spdlog_header_only
    cereal::cereal
    Boost::asio
    JPEG::JPEG
    lz4::lz4
    Ceres::ceres
    gRPC::grpc++
    imgui::imgui
    glm::glm
)

target_compile_options(stargazer_viewer
    PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:Clang>:$<$<CONFIG:RELEASE>:-O2>>
    $<$<CXX_COMPILER_ID:Clang>:$<$<CONFIG:DEBUG>:-g -O0>>
    $<$<BOOL:${MSVC}>:/bigobj>)

target_sources(stargazer_viewer
    PRIVATE
)

set_target_properties(stargazer_viewer PROPERTIES
    CXX_STANDARD 20)

# gRPC server standalone application
add_executable(stargazer_grpc_server
    ${CMAKE_CURRENT_LIST_DIR}/grpc_server_app.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gui.cpp
    ${CMAKE_CURRENT_LIST_DIR}/viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/views.cpp
    ${CMAKE_CURRENT_LIST_DIR}/render3d.cpp
)

add_dependencies(stargazer_grpc_server
    stargazer
    stargazer_sensor_grpc_proto
)

target_include_directories(stargazer_grpc_server
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_definitions(stargazer_grpc_server
    PUBLIC
    $<$<BOOL:${MSVC}>:_USE_MATH_DEFINES>
    $<$<BOOL:${MSVC}>:GLOG_NO_ABBREVIATED_SEVERITIES>
)

target_link_directories(stargazer_grpc_server
    PRIVATE
    ${VCPKG_ROOT}/installed/x64-linux/lib/
)

target_link_libraries(stargazer_grpc_server
    PRIVATE
    stargazer
    stargazer_sensor_grpc_proto
    opencv_imgproc
    opencv_highgui
    opencv_imgcodecs
    $<$<NOT:$<BOOL:${WIN32}>>:vulkan>
    glfw
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
    $<$<NOT:$<BOOL:${WIN32}>>:dl>
    spdlog::spdlog_header_only
    cereal::cereal
    Boost::asio
    gRPC::grpc++
    glm::glm
    imgui::imgui
)

target_compile_options(stargazer_grpc_server
    PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:Clang>:$<$<CONFIG:RELEASE>:-O2>>
    $<$<CXX_COMPILER_ID:Clang>:$<$<CONFIG:DEBUG>:-g -O0>>
    $<$<BOOL:${MSVC}>:/bigobj>)

set_target_properties(stargazer_grpc_server PROPERTIES
    CXX_STANDARD 20)

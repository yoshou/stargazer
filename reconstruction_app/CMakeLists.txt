cmake_minimum_required(VERSION 3.8)

project(reconstruction_app)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/viewer_app.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../lib
    ${CMAKE_CURRENT_LIST_DIR}/../lib/reconstruction
)

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
)

add_dependencies(${PROJECT_NAME}
    stargazer
)

target_link_libraries(${PROJECT_NAME}
    glfw
    GL
    $<$<BOOL:${WIN32}>:GLEW::GLEW>
    $<$<NOT:$<BOOL:${WIN32}>>:GLEW>
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
)

target_compile_options(${PROJECT_NAME}
    PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wno-unused-parameter>
    $<$<CONFIG:RELEASE>:-O2>
    $<$<CONFIG:DEBUG>:-g -O0>)

target_sources(${PROJECT_NAME}
    PRIVATE
)

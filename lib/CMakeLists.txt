cmake_minimum_required(VERSION 3.8)

project(stargazer)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction
    ${CMAKE_CURRENT_LIST_DIR}/calibration
)

add_library(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/camera_info.cpp
    ${CMAKE_CURRENT_LIST_DIR}/multiview_point_data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction/triangulation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction/correspondance.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calibration/bundle_adjust_data.cpp
)

add_dependencies(${PROJECT_NAME}
    json
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    pthread
    yaml-cpp
    stdc++fs
)

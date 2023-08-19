cmake_minimum_required(VERSION 3.8)

add_library(stargazer
    ${CMAKE_CURRENT_LIST_DIR}/camera_info.cpp
    ${CMAKE_CURRENT_LIST_DIR}/multiview_point_data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction/triangulation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction/correspondance.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calibration/bundle_adjust_data.cpp
)

target_include_directories(stargazer
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction
    ${CMAKE_CURRENT_LIST_DIR}/calibration
)

target_link_libraries(stargazer
    opencv_imgproc
    pthread
    yaml-cpp
    stdc++fs
)

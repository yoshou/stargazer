include(FetchContent)

FetchContent_Declare(
    mplus-fonts
    GIT_REPOSITORY https://github.com/coz-m/MPLUS_FONTS.git
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/mplus"
)

FetchContent_MakeAvailable(mplus-fonts)
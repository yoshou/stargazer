include(FetchContent)

FetchContent_Populate(
    mplus-fonts
    GIT_REPOSITORY https://github.com/coz-m/MPLUS_FONTS.git
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/mplus"
)

# Download Font Awesome 6 Free Solid TTF at configure time
set(FA6_TTF "${CMAKE_CURRENT_LIST_DIR}/fontawesome/fa-solid-900.ttf")
if(NOT EXISTS "${FA6_TTF}")
    message(STATUS "Downloading Font Awesome 6 Free Solid TTF...")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/fontawesome")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.7.2/webfonts/fa-solid-900.ttf"
        "${FA6_TTF}"
        EXPECTED_HASH SHA256=af19d135d3a935b3ebfbd80320716ffe1202052c5f68dc2c5f1abc57005ac605
        SHOW_PROGRESS
    )
endif()
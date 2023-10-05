cmake_minimum_required(VERSION 2.7.0...3.27.0)
project(stereovision_disparity VERSION 0.1.0 LANGUAGES C CXX)

# include(CTest)
# enable_testing()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(PCL 1.3 REQUIRED)

pkg_check_modules(yaml_cpp REQUIRED IMPORTED_TARGET yaml-cpp)
pkg_check_modules(jsoncpp REQUIRED IMPORTED_TARGET jsoncpp)
pkg_check_modules(fmt REQUIRED IMPORTED_TARGET fmt)

# include_directories(include)
# add_definitions(
#     -DBUILD_opencv_vtk=OFF
#     -DWITH_VTK=OFF
# )

add_executable(${CMAKE_PROJECT_NAME}
    main.cpp

    src/ReadStereoFS.cpp
    src/CamModify.cpp
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE 
    include

    ${OpenCV_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    PkgConfig::yaml_cpp
    PkgConfig::jsoncpp
    PkgConfig::fmt
)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE 
    "CONFIG_DIR_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/config/\""
    "DATA_DIR_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/data/\""
    ${PCL_DEFINITIONS}
)

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)

cmake_minimum_required(VERSION 3.5)

project(LBM_c LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXECUTABLE_NAME "LBM_c")

set(homeDir "/home/geotyper")

find_package(SFML 2.5.1 COMPONENTS graphics)
set (INCLUDES ${INCLUDES} ${SFML_INCLUDE_DIRS})
set (LIBS ${LIBS} ${SFML_LIBRARIES})

set (IMGUI_DIR ${homeDir}/CPP_lib/imGui/imgui-master)
find_package(ImGui-SFML REQUIRED)


include_directories(${IMGUISFML_DIR})
include_directories(${IMGUI_DIR})
include_directories(${INCLUDE_SDIR})
link_directories(${INCLUDE_SDIR})


set(Head_files
    math/cpVect.h
    math/cpBB.h
    math/chipmunk_types.h
    math/cpTransform.h
    math/tinycolormap.hpp
    kernel.cuh
    display.hpp
    mainloop.hpp


    )

set(Source_files

    display.cpp
    mainloop.cpp
    kernel.cu
    main.cpp

    )

set(ALL_FILES
    ${Head_files}
    ${Source_files}
    ${INCLUDES}
    )


find_package(CUDA REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_61,code=sm_61
   )


cuda_add_executable(${EXECUTABLE_NAME} ${ALL_FILES})

target_include_directories(${EXECUTABLE_NAME} PRIVATE ${INCLUDES} )

target_link_libraries(${EXECUTABLE_NAME} sfml-graphics)
target_link_libraries(${EXECUTABLE_NAME} ImGui-SFML::ImGui-SFML)

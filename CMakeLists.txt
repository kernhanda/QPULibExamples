cmake_minimum_required(VERSION 3.4)
project(QPULibExamples CXX)
set(CMAKE_CXX_STANDARD 14)

include(ExternalProject)

option(ENABLE_DEBUG "Output debug information from QPULib" OFF)
option(ENABLE_QPU "Enable QPU (rpi only), otherwise emulate" OFF)

ExternalProject_Add(QPULib-project
    PREFIX 				${CMAKE_BINARY_DIR}/QPULib
    SOURCE_DIR 			${CMAKE_SOURCE_DIR}/lib/QPULib
    GIT_REPOSITORY		https://github.com/kernhanda/QPULib.git
    UPDATE_COMMAND 		git pull -f https://github.com/kernhanda/QPULib.git development
    GIT_TAG development
    # STEP_TARGETS 		build
    # EXCLUDE_FROM_ALL	TRUE
    TIMEOUT 			30		#Timeout for downloads, in seconds
        CMAKE_ARGS
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_FIND_ROOT_PATH=${CMAKE_FIND_ROOT_PATH}
            -DENABLE_DEBUG=${ENABLE_DEBUG}
            -DENABLE_QPU=${ENABLE_QPU}
            -DBUILD_EXAMPLES=OFF
)

add_executable(Dot Dot.cpp)
target_link_libraries(Dot qpu.a)
add_dependencies(Dot QPULib-project)

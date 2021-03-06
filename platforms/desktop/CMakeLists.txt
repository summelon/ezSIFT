cmake_minimum_required(VERSION 2.8)

set(EZSIFT ezsift)
project(${EZSIFT})

option(EZSIFT_BUILD_EXAMPLES "Build with examples" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
endforeach()

set(EZSIFT_ROOT ${PROJECT_SOURCE_DIR}/../../)
set(EZSIFT_INCLUDE_DIR ${EZSIFT_ROOT}/include)
set(EZSIFT_SRC_DIR ${EZSIFT_ROOT}/src)

file(GLOB_RECURSE SRC RELATIVE ${PROJECT_SOURCE_DIR} "${EZSIFT_SRC_DIR}/*.cpp")
file(GLOB_RECURSE HEADERS RELATIVE ${PROJECT_SOURCE_DIR} "${EZSIFT_SRC_DIR}/*.h" "${EZSIFT_INCLUDE_DIR}/*.h")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_library(${EZSIFT} ${SRC})
set_property(TARGET ${EZSIFT} PROPERTY POSITION_INDEPENDENT_CODE ON)

include_directories(
    ${EZSIFT_INCLUDE_DIR}
)

target_include_directories(${EZSIFT} PUBLIC ${EZSIFT_SRC_DIR})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


if(EZSIFT_BUILD_EXAMPLES)
  add_subdirectory(${EZSIFT_ROOT}/examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples/)
endif()

OPTION(DEFINE_DEBUG
  "Build the project using debugging code"
  OFF)
IF(DEFINE_DEBUG)
  MESSAGE("Adding Debug flag...")
  ADD_DEFINITIONS(-DDEBUG)
  SET(CMAKE_BUILD_TYPE Debug)
  MESSAGE("Build type is " ${CMAKE_BUILD_TYPE})
ENDIF(DEFINE_DEBUG)

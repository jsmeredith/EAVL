cmake_minimum_required(VERSION 2.8)
project(EAVL)

#-----------------------------------------------------------------------------
# Run all the CMake configuration tests for our build environment
#-----------------------------------------------------------------------------
if(NOT EAVL_EXTERNALLY_CONFIGURED)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables."
  )
  set(EXECUTABLE_OUTPUT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
      ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries"
  )
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
      ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries."
  )
endif(NOT EAVL_EXTERNALLY_CONFIGURED)

#-----------------------------------------------------------------------------
# Run all the CMake configuration tests for our build environment
#-----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${EAVL_SOURCE_DIR}/cmake)
include(${CMAKE_MODULE_PATH}/ConfigureChecks.cmake)
include(${CMAKE_MODULE_PATH}/EAVLMacros.cmake)

#-----------------------------------------------------------------------------
# Find Boost
#-----------------------------------------------------------------------------
option(Boost_USE_STATIC_LIBS "Use static Boost libraries" OFF)
SET(Boost_ADDITIONAL_VERSIONS "1.48" "1.48.0" "1.49" "1.49.0")
SET(Boost_DEBUG ON)
find_package(Boost 1.48)
link_directories(${Boost_LIBRARY_DIRS})

#-----------------------------------------------------------------------------
# Find CUDA
#-----------------------------------------------------------------------------
option (BUILD_CUDA "Build CUDA support" ON)
IF (BUILD_CUDA)
  find_package(CUDA)
  IF (CUDA_FOUND)
    SET(HAVE_CUDA 1)
    include_directories(${CUDA_INCLUDE_DIRS})
  ENDIF (CUDA_FOUND)

  IF(NOT WIN32)
      SET(CUDA_NVCC_FLAGS_USER "-gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20" CACHE STRING "CUDA NVCC compile flags")
  ELSE (NOT WIN32)
      SET(CUDA_NVCC_FLAGS_USER "--gpu-architecture=compute_20" CACHE STRING "CUDA NVCC compile flags")
  ENDIF (NOT WIN32)
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_USER} CACHE INTERNAL "CUDA NVCC compile flags" FORCE)

ENDIF (BUILD_CUDA)

#-----------------------------------------------------------------------------
# setup a global variable that we will add all libraries to.
# For export of targets, so that other projects can pick them up cleanly
#-----------------------------------------------------------------------------
INIT_GLOBAL_LIST(EAVL_EXPORTED_LIBS "")

#-----------------------------------------------------------------------------
# Include directories
#-----------------------------------------------------------------------------
set(EAVL_INCLUDE_DIRECTORIES
  ${EAVL_BINARY_DIR}
  ${EAVL_SOURCE_DIR}/config
  ${EAVL_SOURCE_DIR}/src/common
  ${EAVL_SOURCE_DIR}/src/exporters
  ${EAVL_SOURCE_DIR}/src/filters
  ${EAVL_SOURCE_DIR}/src/fonts
  ${EAVL_SOURCE_DIR}/src/importers
  ${EAVL_SOURCE_DIR}/src/math
  ${EAVL_SOURCE_DIR}/src/operations
  ${EAVL_SOURCE_DIR}/src/rendering
)
IF (HAVE_CUDA)
  list(APPEND EAVL_INCLUDE_DIRECTORIES
    "${CUDA_INCLUDE_DIRS}"
    "${EAVL_SOURCE_DIR}/src/operations"
  )
ENDIF (HAVE_CUDA)

include_directories(${EAVL_INCLUDE_DIRECTORIES})

#-----------------------------------------------------------------------------
# setup src
#-----------------------------------------------------------------------------
add_subdirectory(src)

#-----------------------------------------------------------------------------
# Generate the eavlConfig.h file containing settings needed for compilation
#-----------------------------------------------------------------------------
CONFIGURE_FILE (cmake/eavlConfig.h.in ${EAVL_BINARY_DIR}/eavlConfig.h @ONLY)

#-----------------------------------------------------------------------------
# Export all exported targets to the build tree for use by parent project
#-----------------------------------------------------------------------------
EXPORT (
  TARGETS
    ${EAVL_EXPORTED_LIBS}
  FILE
    eavl-targets.cmake
)

#-----------------------------------------------------------------------------
# Configure the EAVL-config.cmake file for the build directory
#-----------------------------------------------------------------------------
set(EAVL_INCLUDES_BUILD_TIME ${EAVL_INCLUDE_DIRECTORIES})

configure_file(
  ${EAVL_SOURCE_DIR}/cmake/eavl-config.cmake.build.in 
  ${EAVL_BINARY_DIR}/eavl-config.cmake @ONLY
)

#-----------------------------------------------------------------------------
# Testing
#-----------------------------------------------------------------------------
option(BUILD_TESTING "Enable testing" ON)
IF(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
ENDIF(BUILD_TESTING)


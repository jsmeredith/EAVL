cmake_minimum_required(VERSION 2.8)
project(EAVL)

#-----------------------------------------------------------------------------
# Run all the CMake configuration tests for our build environment
#-----------------------------------------------------------------------------
INCLUDE (cmake/ConfigureChecks.cmake)
SET(CMAKE_MODULE_PATH ${EAVL_SOURCE_DIR}/CMake)

#-----------------------------------------------------------------------------
# Find Boost
#-----------------------------------------------------------------------------
option(Boost_USE_STATIC_LIBS "Use static Boost libraries" OFF)
SET(Boost_ADDITIONAL_VERSIONS "1.48" "1.48.0" "1.49" "1.49.0")
SET(Boost_DEBUG ON)
find_package(Boost 1.48 REQUIRED)
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

  option (CUDA_OLDGPU "Build CUDA : support compute capability pre-2.0 GPUs" OFF)
  IF (CUDA_OLDGPU) 
    SET(HAVE_OLD_GPU 1)
    SET(CUDA_NVCC_FLAGS_USER "--gpu-architecture compute_13" CACHE STRING "CUDA NVCC compile flags")
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_USER} CACHE INTERNAL "CUDA NVCC compile flags" FORCE)
    MESSAGE("Pre compute 2 GPUs are not fully supported. Not all functions have been implemented and some link errors may occur")
  ELSE (CUDA_OLDGPU) 
    SET(CUDA_NVCC_FLAGS_USER "--gpu-architecture compute_20" CACHE STRING "CUDA NVCC compile flags")
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_USER} CACHE INTERNAL "CUDA NVCC compile flags" FORCE)
  ENDIF (CUDA_OLDGPU) 

ENDIF (BUILD_CUDA)

#-----------------------------------------------------------------------------
# Include directories
#-----------------------------------------------------------------------------
include_directories(
  ${EAVL_BINARY_DIR}
  ${EAVL_SOURCE_DIR}/config
  ${EAVL_SOURCE_DIR}/src/math
  ${EAVL_SOURCE_DIR}/src/common
)
IF (HAVE_CUDA)
  include_directories(
    "${CUDA_INCLUDE_DIRS}"
    "${EAVL_SOURCE_DIR}/src/operations"
  )
ENDIF (HAVE_CUDA)

#-----------------------------------------------------------------------------
# setup src
#-----------------------------------------------------------------------------
add_subdirectory(src)


#-----------------------------------------------------------------------------
# Generate the eavlConfig.h file containing settings needed by compilation
#-----------------------------------------------------------------------------
CONFIGURE_FILE (cmake/eavlConfig.h.in ${EAVL_BINARY_DIR}/eavlConfig.h @ONLY)

#-----------------------------------------------------------------------------
# Testing
#-----------------------------------------------------------------------------
option(BUILD_TESTING "Enable testing" ON)
IF(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
ENDIF(BUILD_TESTING)


SET(EAVL_FILTERS_SRCS
  eavl2DGraphLayoutForceMutator.cpp
  eavlBoxMutator.cpp
  eavlElevateMutator.cpp
  eavlExternalFaceMutator.cpp
  eavlSubsetMutator.cpp
  eavlTesselate2DFilter.cpp
  eavlThresholdMutator.cpp
)

SET(EAVL_FILTERS_CUDA_SRCS 
  eavl3X3AverageMutator.cu
  eavlBinaryMathMutator.cu
  eavlCellToNodeRecenterMutator.cu
  eavlIsosurfaceFilter.cu
  eavlPointDistanceFieldFilter.cu
  eavlScalarBinFilter.cu
  eavlSurfaceNormalMutator.cu
  eavlTransformMutator.cu
  eavlUnaryMathMutator.cu
  eavlSelectionMutator.cu
)

IF (HAVE_CUDA) 
  # set this, otherwise C_SOURCES is empty in Visual Studio  
  set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF) 

  cuda_add_library(eavl_filters_cu
    ${EAVL_FILTERS_CUDA_SRCS}
  )
  ADD_GLOBAL_LIST(EAVL_EXPORTED_LIBS eavl_filters_cu)

  add_library(eavl_filters
    ${EAVL_FILTERS_SRCS}
  )
  target_link_libraries(eavl_filters eavl_filters_cu)
  ADD_GLOBAL_LIST(EAVL_EXPORTED_LIBS eavl_filters)

ELSE (HAVE_CUDA)
  set_source_files_properties(${EAVL_FILTERS_CUDA_SRCS} PROPERTIES LANGUAGE CXX)
  IF (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    set_source_files_properties(${EAVL_FILTERS_CUDA_SRCS} PROPERTIES COMPILE_FLAGS "-x c++")
  ENDIF (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  add_library(eavl_filters 
    ${EAVL_FILTERS_SRCS}
    ${EAVL_FILTERS_CUDA_SRCS}
  )
ENDIF (HAVE_CUDA) 

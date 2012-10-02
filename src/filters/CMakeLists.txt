SET(EAVL_FILTERS_SRCS
  eavlElevateMutator.cpp
  eavlExternalFaceMutator.cpp
  eavlTesselate2DFilter.cpp
  eavlThresholdMutator.cpp
)

IF (HAVE_CUDA) 
  SET(EAVL_FILTERS_CUDA_SRCS 
    eavl3X3AverageMutator.cu
    eavlIsosurfaceFilter.cu
    eavlSurfaceNormalMutator.cu
    eavlTransformMutator.cu
    eavlBinaryMathMutator.cu
  )
  cuda_add_library(eavl_filters
    ${EAVL_FILTERS_SRCS}
    ${EAVL_FILTERS_CUDA_SRCS}
  )
ELSE (HAVE_CUDA)
  add_library(eavl_filters 
    ${EAVL_FILTERS_SRCS}
  )
ENDIF (HAVE_CUDA) 


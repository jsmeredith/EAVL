SET(EAVL_RENDERING_SRCS
  eavlColor.cpp
)

add_library(eavl_rendering 
  ${EAVL_RENDERING_SRCS}
)

ADD_GLOBAL_LIST(EAVL_EXPORTED_LIBS eavl_rendering)

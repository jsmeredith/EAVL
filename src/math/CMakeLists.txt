SET(EAVL_MATH_SRCS
  eavlMatrix4x4.cpp
  eavlPoint3.cpp
  eavlVector3.cpp
)

add_library(eavl_math 
  ${EAVL_MATH_SRCS}
)
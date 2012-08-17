SET(EAVL_EXPORTERS_SRCS
  eavlPNMExporter.cpp
  eavlVTKExporter.cpp
)

add_library(eavl_exporters 
  ${EAVL_EXPORTERS_SRCS}
)
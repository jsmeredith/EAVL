SET(EAVL_COMMON_SRCS
  eavlArray.cpp
  eavlAtomicProperties.cpp
  eavlCellComponents.cpp
  eavlCellSet.cpp
  eavlCellSetExplicit.cpp
  eavlCompositor.cpp
  eavlCoordinates.cpp
  eavlCUDA.cpp
  eavlDataSet.cpp
  eavlExecutor.cpp
  eavlFlatArray.cpp
  eavlLogicalStructure.cpp
  eavlNewIsoTables.cpp
  eavlOperation.cpp
  eavlTimer.cpp
  eavlUtility.cpp
)

add_library(eavl_common 
  ${EAVL_COMMON_SRCS}
)

ADD_GLOBAL_LIST(EAVL_EXPORTED_LIBS eavl_common)
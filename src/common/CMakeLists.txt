SET(EAVL_COMMON_SRCS
  eavlArray.cpp
  eavlAtomicProperties.cpp
  eavlCellComponents.cpp
  eavlCellSetExplicit.cpp
  eavlCUDA.cpp
  eavlDataSet.cpp
  eavlExecutor.cpp
  eavlNewIsoTables.cpp
  eavlOperation.cpp
  eavlTimer.cpp
)

add_library(eavl_common 
  ${EAVL_COMMON_SRCS}
)
SET(EAVL_IMPORTERS_SRCS
  eavlImporterFactory.cpp
  eavlBOVImporter.cpp
  eavlImporterFactory.cpp
  eavlMADNESSImporter.cpp
  eavlPDBImporter.cpp
  eavlVTKImporter.cpp
  eavlCurveImporter.cpp
  eavlPNGImporter.cpp
  eavlLAMMPSDumpImporter.cpp
)

#-------------------------------
# ADIOS
#-------------------------------
IF(EAVL_HAVE_ADIOS)
  SET(EAVL_ADIOS_SRCS
    ADIOSFileObject.cpp
    eavlADIOSImporter.cpp
    eavlPixieImporter.cpp
    # seems to depend on ADIOS
    eavlPixieImporter.cpp
    eavlXGCImporter.cpp
  )
ENDIF(EAVL_HAVE_ADIOS)

#-------------------------------
# SILO
#-------------------------------
IF(EAVL_HAVE_SILO)
  SET(EAVL_SILO_SRCS
    eavlSiloImporter.cpp
  )
ENDIF(EAVL_HAVE_SILO)

#-------------------------------
# NETCDF
#-------------------------------
IF(EAVL_HAVE_NETCDF)
  SET(EAVL_NETCDF_SRCS
    eavlNetCDFDecomposingImporter.cpp
    eavlNetCDFImporter.cpp
  )
ENDIF(EAVL_HAVE_NETCDF)

#-------------------------------
# CHIMERA
#-------------------------------
IF(EAVL_HAVE_CHIMERA)
  SET(EAVL_CHIMERA_SRCS
  eavlChimeraImporter.cpp
  )
ENDIF(EAVL_HAVE_CHIMERA)

#-------------------------------
# lib
#-------------------------------
add_library(eavl_importers 
  ${EAVL_IMPORTERS_SRCS}
  ${EAVL_VTK_SRCS}
  ${EAVL_ADIOS_SRCS}
  ${EAVL_SILO_SRCS}
  ${EAVL_NETCDF_SRCS}
  ${EAVL_CHIMERA_SRCS}
)

ADD_GLOBAL_LIST(EAVL_EXPORTED_LIBS eavl_importers)
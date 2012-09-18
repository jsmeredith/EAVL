
QT       -= core gui

TARGET = testimport
TEMPLATE = app
CONFIG += debug console


SOURCES += testimport.cpp

EAVLROOT = ..

HEADERS  += $$files($$EAVLROOT/src/*/*.h)

DEPENDPATH += $$EAVLROOT/config $$EAVLROOT/src/common $$EAVLROOT/src/importers $$EAVLROOT/src/filters $$EAVLROOT/src/exporters $$EAVLROOT/src/math $$EAVLROOT/src/rendering
INCLUDEPATH += $$EAVLROOT/config $$EAVLROOT/src/common $$EAVLROOT/src/importers $$EAVLROOT/src/filters $$EAVLROOT/src/exporters $$EAVLROOT/src/math $$EAVLROOT/src/rendering

win32 {
  LIBS += -L$$EAVLROOT/Debug/lib -L$$EAVLROOT/../eavl-build-desktop/debug/lib -leavl
  #POST_TARGETDEPS += $$EAVLROOT/Debug/lib/libeavl.a
}
unix {
  LIBS += -L$$EAVLROOT/lib -leavl
  POST_TARGETDEPS += $$EAVLROOT/lib/libeavl.a
}

!include($$EAVLROOT/config/make-dependencies) {
  INCLUDEPATH += $$EAVLROOT/config-simple
}

HOST = $$system(hostname)
SYS = $$system(uname -s)

!equals(NETCDF, no) {
  INCLUDEPATH += $$NETCDF/include
  LIBS += $$NETCDF_LDFLAGS $$NETCDF_LIBS
}

!equals(HDF5, no) {
  INCLUDEPATH += $$HDF5/include
  LIBS += $$HDF5_LDFLAGS $$HDF5_LIBS
}

!equals(CUDA, no) {
  INCLUDEPATH += $$CUDA/include
  LIBS += $$CUDA_LDFLAGS $$CUDA_LIBS
}

!equals(SILO, no) {
  INCLUDEPATH += $$SILO/include
  LIBS += $$SILO_LDFLAGS $$SILO_LIBS
}

!equals(ADIOS, no) {
  INCLUDEPATH += $$ADIOS/include
  LIBS += $$ADIOS_LDFLAGS $$ADIOS_LIBS
}

!equals(SZIP, no) {
  INCLUDEPATH += $$SZIP/include
  LIBS += $$SZIP_LDFLAGS $$SZIP_LIBS
}

!equals(ZLIB, no) {
  INCLUDEPATH += $$ZLIB/include
  LIBS += $$ZLIB_LDFLAGS $$ZLIB_LIBS
}

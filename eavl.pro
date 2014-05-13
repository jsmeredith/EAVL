#-------------------------------------------------
#
# Project created by QtCreator 2011-03-08T13:20:12
#
#-------------------------------------------------

QT       -= core gui

TARGET = lib/eavl
TEMPLATE = lib
CONFIG += staticlib

# hack in the CUDA files as c plus plus files
QMAKE_EXT_CPP = .cu
QMAKE_CXXFLAGS += -x c++

INCLUDEPATH += config/ src/common src/executor src/exporters src/fonts src/functors src/filters src/importers src/math src/operations src/rendering

# we don't have the benefit of running configure
# point to this pre-generated eavlConfig.h with
# that assumes no dependencies or 3rd party libs
INCLUDEPATH += config-simple

SOURCES += \
    src/common/eavlArray.cpp \
    src/common/eavlAtomicProperties.cpp \
    src/common/eavlCUDA.cpp \
    src/common/eavlCellComponents.cpp \
    src/common/eavlCellSet.cpp \
    src/common/eavlCellSetExplicit.cpp \
    src/common/eavlCoordinates.cpp \
    src/common/eavlDataSet.cpp \
    src/common/eavlExecutor.cpp \
    src/common/eavlFlatArray.cpp \
    src/common/eavlLogicalStructure.cpp \
    src/common/eavlNewIsoTables.cpp \
    src/common/eavlOperation.cpp \
    src/common/eavlTimer.cpp \
    src/common/eavlUtility.cpp \
    src/exporters/eavlPNMExporter.cpp \
    src/exporters/eavlVTKExporter.cpp \
    src/filters/eavl3X3AverageMutator.cu \
    src/filters/eavlBinaryMathMutator.cu \
    src/filters/eavlCellToNodeRecenterMutator.cu \
    src/filters/eavlElevateMutator.cpp \
    src/filters/eavlExternalFaceMutator.cpp \
    src/filters/eavlIsosurfaceFilter.cu \
    src/filters/eavlScalarBinFilter.cu \
    src/filters/eavlSurfaceNormalMutator.cu \
    src/filters/eavlTesselate2DFilter.cpp \
    src/filters/eavlThresholdMutator.cpp \
    src/filters/eavlTransformMutator.cu \
    src/filters/eavlUnaryMathMutator.cu \
    src/fonts/Liberation2Mono.cpp \
    src/fonts/Liberation2Sans.cpp \
    src/fonts/Liberation2Serif.cpp \
    src/fonts/eavlBitmapFont.cpp \
    src/fonts/eavlBitmapFontFactory.cpp \
    src/importers/eavlBOVImporter.cpp \
    src/importers/eavlCurveImporter.cpp \
    src/importers/eavlImporterFactory.cpp \
    src/importers/eavlMADNESSImporter.cpp \
    src/importers/eavlPDBImporter.cpp \
    src/importers/eavlPNGImporter.cpp \
    src/importers/eavlVTKImporter.cpp \
    src/math/eavlMatrix4x4.cpp \
    src/math/eavlPoint3.cpp \
    src/math/eavlVector3.cpp \
    src/rendering/eavlColor.cpp

HEADERS += $$files(src/common/*.h) $$files(src/exporters/*.h) $$files(src/filters/*.h) $$files(src/importers/*.h) $$files(src/math/*.h) $$files(src/operations/*.h) $$files(src/rendering/*.h)

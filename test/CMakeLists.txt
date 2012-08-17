#-----------------------------------------------------------------------------
# Tests on systems with batch systems might need a extra launch command
#-----------------------------------------------------------------------------
SET(EAVL_TEST_RUN_COMMAND CACHE STRING "Supply a command such as 'aprun' to execute tests")
SET(EAVL_TEST_RUN_ARGS CACHE STRING "Supply args to the run command")

#-----------------------------------------------------------------------------
# includes
#-----------------------------------------------------------------------------
include_directories(
  ${EAVL_SOURCE_DIR}/src/importers
  ${EAVL_SOURCE_DIR}/src/exporters
  ${EAVL_SOURCE_DIR}/src/filters
)

#-----------------------------------------------------------------------------
# test import
#-----------------------------------------------------------------------------
add_executable(
  testimport
  testimport.cpp
)
target_link_libraries(testimport eavl_importers eavl_filters eavl_common)

set(datafiles
    curv_cube.vtk
    curv_x.vtk
    curv_xy.vtk
    curv_xz.vtk
    curv_y.vtk
    curv_yz.vtk
    curv_z.vtk
    poly_1d_in_3d.vtk
    poly_2d_in_3d.vtk
    poly_sphere.vtk
    poly_verts_and_lines.vtk
    poly_verts.vtk
    rect_cube.vtk
    rect_x.vtk
    rect_xy.vtk
    rect_xz.vtk
    rect_y.vtk
    rect_yz.vtk
    rect_z.vtk
    ucd_1d_in_3d.vtk
    ucd_2d_xy.vtk
    ucd_cube.vtk
    ucd_sphere.vtk
    ucd_tets.vtk
)

foreach(datafile ${datafiles})
  add_test(
    NAME testimport_${datafile}
    COMMAND
      ${EAVL_TEST_RUN_COMMAND} ${EAVL_TEST_RUN_ARGS}
      $<TARGET_FILE:testimport>
      "${EAVL_SOURCE_DIR}/data/${datafile}"
  )
endforeach(datafile)

#-----------------------------------------------------------------------------
# test iso
#-----------------------------------------------------------------------------
add_executable(
  testiso
  testiso.cpp
)
target_link_libraries(testiso eavl_exporters eavl_importers eavl_filters eavl_common)

set(datafiles
  rect_cube.vtk
  curv_cube.vtk
  ucd_cube.vtk
  ucd_sphere.vtk
  ucd_tets.vtk
)

foreach(datafile ${datafiles})
  add_test(
    NAME testiso_${datafile}
    COMMAND
      ${EAVL_TEST_RUN_COMMAND} ${EAVL_TEST_RUN_ARGS}
      $<TARGET_FILE:testiso>
      3.5 nodal "${EAVL_SOURCE_DIR}/data/${datafile}"
  )
endforeach(datafile)

#-----------------------------------------------------------------------------
# test normal
#-----------------------------------------------------------------------------
add_executable(
  testnormal
  testnormal.cpp
)
target_link_libraries(testnormal eavl_exporters eavl_importers eavl_filters eavl_common)

set(datafiles
  curv_cube.vtk
  curv_xy.vtk
  curv_xz.vtk
  curv_yz.vtk
  poly_2d_in_3d.vtk
  poly_sphere.vtk
  rect_cube.vtk
  rect_xy.vtk
  rect_xz.vtk
  rect_yz.vtk
  ucd_2d_xy.vtk
  ucd_cube.vtk
  ucd_sphere.vtk
  ucd_tets.vtk
)

foreach(datafile ${datafiles})
  add_test(
    NAME testnormal_${datafile}
    COMMAND
      ${EAVL_TEST_RUN_COMMAND} ${EAVL_TEST_RUN_ARGS}
      $<TARGET_FILE:testnormal>
      "${EAVL_SOURCE_DIR}/data/${datafile}"
  )
endforeach(datafile)

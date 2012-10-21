#-----------------------------------------------------------------------------
# Tests on systems with batch systems might need a extra launch command
#-----------------------------------------------------------------------------
SET(EAVL_TEST_RUN_COMMAND CACHE STRING "Supply a command such as 'aprun' to execute tests")
SET(EAVL_TEST_RUN_ARGS CACHE STRING "Supply args to the run command")

#-----------------------------------------------------------------------------
# Find diff, we'll use diff to display errors in text comparison
#-----------------------------------------------------------------------------
find_package(Diff)

#-----------------------------------------------------------------------------
# includes
#-----------------------------------------------------------------------------
include_directories(
  ${EAVL_SOURCE_DIR}/src/importers
  ${EAVL_SOURCE_DIR}/src/exporters
  ${EAVL_SOURCE_DIR}/src/filters
)

#-----------------------------------------------------------------------------
# define macro for testing to compare stdout against baseline text file
#-----------------------------------------------------------------------------
include (CMakeParseArguments)

MACRO(ADD_TEXT_TEST)
  CMAKE_PARSE_ARGUMENTS(
    TEST 
    ""
    "NAME;COMMAND;BASELINE"
    "ARGSLIST" 
    ${ARGN}
  ) 
#  message("TEST_DEFAULT_ARGS is ${TEST_DEFAULT_ARGS} from ${ARGN}")
#  message("TEST_NAME is ${TEST_NAME}")
#  message("TEST_COMMAND is ${TEST_COMMAND}")
#  message("TEST_ARGSLIST is ${TEST_ARGSLIST}")
#  message("TEST_BASELINE is ${TEST_BASELINE}")

  # if tests have to be run using aprun or srun or some command other
  # than the test executable itself, then we set the command and pass
  # the test as an argument

  if (EAVL_TEST_RUN_COMMAND) 
    set(TEST_ARGUMENTS "${EAVL_TEST_RUN_ARGS}" "${TEST_COMMAND}" "${TEST_ARGSLIST}" "${TEST_BASELINE}")
    set(TEST_COMMAND "${EAVL_TEST_RUN_COMMAND}")
  endif(EAVL_TEST_RUN_COMMAND)

  # We call a script which executes the text, captures the output
  # and compares it to the baseline.
  #
  # When we call the script, all our arguments get serialized and quoted by cmake,
  # this breaks the args we pass into the test itself (especially when spaces are present).
  # So we first generate one big string containing all our args, plus test command
  # and all the other info needed by the script - turn it all into one big list (separator ;)
  # containing keywords we can use in PARSE_ARGUMENTS. We pass the whole lot
  # to cmake execute script using -D to hold the big string. 
  # The script uses the parse arguments macro to regenerate the original set of args/commands etc
  # we can extend it by adding keywords at will, and modifying runTest.cmake to find them.

  set(arg_string 
    COMMAND         ${TEST_COMMAND} 
    ARGSLIST        ${TEST_ARGSLIST} 
    OUTPUT          tmpfile.txt 
    BASELINE        ${TEST_BASELINE}
    DIFF_EXECUTABLE ${DIFF_EXECUTABLE}
  )

  # escape all spaces before passing to args
  string(REPLACE " " "\\ " TEST_ARGSLIST2 "${arg_string}" )
  # escape all ; before passing to args
  string(REPLACE ";" "\\;" TEST_ARGSLIST3 "${TEST_ARGSLIST2}" )

  add_test( 
    NAME 
      ${TEST_NAME}
    COMMAND
      ${CMAKE_COMMAND} 
      -D SCRIPTARGS=${TEST_ARGSLIST3}
      -P ${EAVL_SOURCE_DIR}/cmake/runTest.cmake 
  )
ENDMACRO(ADD_TEXT_TEST)

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
  ADD_TEXT_TEST(
    NAME
      "testimport_${datafile}"
    COMMAND
      "$<TARGET_FILE:testimport>"
    ARGSLIST
      "${EAVL_SOURCE_DIR}/data/${datafile}"
    BASELINE
      "${CMAKE_CURRENT_SOURCE_DIR}/baseline/testimport/${datafile}.out"
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
  ADD_TEXT_TEST(
    NAME 
      testiso_${datafile}
    COMMAND
      "$<TARGET_FILE:testiso>"
    ARGSLIST
      3.5 nodal "${EAVL_SOURCE_DIR}/data/${datafile}"
    BASELINE
      "${CMAKE_CURRENT_SOURCE_DIR}/baseline/testiso/${datafile}.out"
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
  ADD_TEXT_TEST(
    NAME
      testnormal_${datafile}
    COMMAND
      "$<TARGET_FILE:testnormal>"
    ARGSLIST
      "${EAVL_SOURCE_DIR}/data/${datafile}"
    BASELINE
      "${CMAKE_CURRENT_SOURCE_DIR}/baseline/testnormal/${datafile}.out"
  )
endforeach(datafile)

#-----------------------------------------------------------------------------
# test math
#-----------------------------------------------------------------------------
add_executable(
  testmath
  testmath.cpp
)
target_link_libraries(testmath eavl_exporters eavl_importers eavl_filters eavl_common)

#-----------------------------------------------------------------------------
# test xform
#-----------------------------------------------------------------------------
add_executable(
  testxform
  testxform.cpp
)
target_link_libraries(testxform eavl_exporters eavl_importers eavl_filters eavl_common)

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
  ADD_TEXT_TEST(
    NAME
      testxform_${datafile}
    COMMAND
      "$<TARGET_FILE:testxform>"
    ARGSLIST
      "${EAVL_SOURCE_DIR}/data/${datafile}"
    BASELINE
      "${CMAKE_CURRENT_SOURCE_DIR}/baseline/testxform/${datafile}.out"
  )
endforeach(datafile)

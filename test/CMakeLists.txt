#-----------------------------------------------------------------------------
# Include convenient argument parsing macro
#-----------------------------------------------------------------------------
include (CMakeParseArguments)

#-----------------------------------------------------------------------------
# Find diff, used to display errors in text comparison of test results
#-----------------------------------------------------------------------------
find_package(Diff)

#-----------------------------------------------------------------------------
# Tests on systems with batch systems might need a extra launch command
#-----------------------------------------------------------------------------
SET(EAVL_TEST_RUN_COMMAND CACHE STRING "Supply a command such as 'aprun' to execute tests")
SET(EAVL_TEST_RUN_ARGS CACHE STRING "Supply args to the run command such as -n 1")

#-----------------------------------------------------------------------------
# includes needed to build tests
#-----------------------------------------------------------------------------
include_directories(
  ${EAVL_SOURCE_DIR}/src/importers
  ${EAVL_SOURCE_DIR}/src/exporters
  ${EAVL_SOURCE_DIR}/src/filters
)

#-----------------------------------------------------------------------------
# simple macro to add a test - sets correct execution command etc
#-----------------------------------------------------------------------------
MACRO(ADD_SIMPLE_TEST)
  CMAKE_PARSE_ARGUMENTS(
    TEST 
    ""
    "NAME;COMMAND"
    "ARGSLIST" 
    ${ARGN}
  ) 

  # if tests have to be run using (for example) aprun instead of the the test executable itself, 
  # then pass the test command as an argument to the other command.
  
  if (EAVL_TEST_RUN_COMMAND) 
    set(TEST_ARGSLIST "${EAVL_TEST_RUN_ARGS}" "${TEST_COMMAND}" "${TEST_ARGSLIST}")
    set(TEST_COMMAND "${EAVL_TEST_RUN_COMMAND}")
  endif(EAVL_TEST_RUN_COMMAND)

  add_test( 
    NAME 
      ${TEST_NAME}
    COMMAND
      ${TEST_COMMAND} ${TEST_ARGSLIST} 
  )
ENDMACRO(ADD_SIMPLE_TEST)

#-----------------------------------------------------------------------------
# define macro for testing to compare stdout against baseline text file
#-----------------------------------------------------------------------------

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
    SET(EAVL_TEST_REMOVE_LAST_N_LINES "0" CACHE STRING "Remove last lines from test output (when extra output causes text comparison failure")
    string(REPLACE " " ";" EAVL_TEST_RUN_ARGS2 ${EAVL_TEST_RUN_ARGS})
    set(TEST_ARGSLIST "${EAVL_TEST_RUN_ARGS2}" "${TEST_COMMAND}" "${TEST_ARGSLIST}")
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
    TRIM_LINES      ${EAVL_TEST_REMOVE_LAST_N_LINES}
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
# List of files used by some tests
#-----------------------------------------------------------------------------

# testimport, testxform, testserialize
set(datafiles_1
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

# testiso
set(datafiles_2
  rect_cube.vtk
  curv_cube.vtk
  ucd_cube.vtk
  ucd_sphere.vtk
  ucd_tets.vtk
)

# testnormal
set(datafiles_3
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

# testdistancefield
set(datafiles_4
  four_points_1d.vtk
  four_points_2d.vtk
  four_points_3d.vtk
)
set(distancefield_params
  "20 -2 7"
  "50 50  -2 7 -2 7"
  "20 20 20  -2 7 -2 7 -2 7"
)

#-----------------------------------------------------------------------------
# test import
#-----------------------------------------------------------------------------
add_executable(
  testimport
  testimport.cpp
)
target_link_libraries(testimport eavl_exporters eavl_importers eavl_filters eavl_common)

foreach(datafile ${datafiles_1})
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

foreach(datafile ${datafiles_2})
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

foreach(datafile ${datafiles_3})
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
# test xform
#-----------------------------------------------------------------------------
add_executable(
  testxform
  testxform.cpp
)
target_link_libraries(testxform eavl_exporters eavl_importers eavl_filters eavl_common)

foreach(datafile ${datafiles_1})
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

#-----------------------------------------------------------------------------
# test serialize
#-----------------------------------------------------------------------------
add_executable(
  testserialize
  testserialize.cpp
)
target_link_libraries(testserialize eavl_importers eavl_filters eavl_common)

foreach(datafile ${datafiles_1})
  ADD_TEXT_TEST(
    NAME
      "testserialize_${datafile}"
    COMMAND
      "$<TARGET_FILE:testserialize>"
    ARGSLIST
      "${EAVL_SOURCE_DIR}/data/${datafile}"
    BASELINE
      "${CMAKE_CURRENT_SOURCE_DIR}/baseline/testserialize/${datafile}.out"
  )
endforeach(datafile)

#-----------------------------------------------------------------------------
# test distance field
#-----------------------------------------------------------------------------
add_executable(
  testdistancefield
  testdistancefield.cpp
)
target_link_libraries(testdistancefield eavl_importers eavl_exporters eavl_filters eavl_common)

list(LENGTH datafiles_4 len1)
math(EXPR len2 "${len1} - 1")

foreach(val RANGE ${len2})
  list(GET datafiles_4          ${val} val1)
  list(GET distancefield_params ${val} val2)
  # convert string of arguments to list
  string(REPLACE " " ";" PARAM ${val2})

  ADD_TEXT_TEST(
    NAME
      "testdistancefield_${val1}"
    COMMAND
      "$<TARGET_FILE:testdistancefield>"
    ARGSLIST
      "${EAVL_SOURCE_DIR}/data/${val1}"
      "${PARAM}"
    BASELINE
      "${CMAKE_CURRENT_SOURCE_DIR}/baseline/testdistancefield/${val1}.out"
  )
endforeach()

#-----------------------------------------------------------------------------
# test data model
#-----------------------------------------------------------------------------
add_executable(
  testdatamodel
  testdatamodel.cpp
)
target_link_libraries(testdatamodel eavl_exporters eavl_filters eavl_common)

ADD_SIMPLE_TEST(
  NAME
    testdatamodel
  COMMAND
    "$<TARGET_FILE:testdatamodel>"
)

#-----------------------------------------------------------------------------
# test math
#-----------------------------------------------------------------------------
add_executable(
  testmath
  testmath.cpp
)
target_link_libraries(testmath eavl_exporters eavl_importers eavl_filters eavl_common)


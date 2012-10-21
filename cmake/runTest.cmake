include (CMakeParseArguments)

CMAKE_PARSE_ARGUMENTS(
  TEST 
  ""
  "COMMAND;BASELINE;OUTPUT;DIFF_EXECUTABLE"
  "ARGSLIST" 
  ${SCRIPTARGS}
) 

# arguments checking
if( NOT TEST_COMMAND )
  message( FATAL_ERROR "Require TEST_COMMAND to be defined" )
endif( NOT TEST_COMMAND )
if( NOT TEST_ARGSLIST )
  message( FATAL_ERROR "Require TEST_ARGSLIST to be defined (form=arg1;arg2;arg3)" )
endif( NOT TEST_ARGSLIST )
if( NOT TEST_OUTPUT )
  message( FATAL_ERROR "Require TEST_OUTPUT to be defined" )
endif( NOT TEST_OUTPUT )
if( NOT TEST_BASELINE )
  message( FATAL_ERROR "Require TEST_BASELINE to be defined" )
endif( NOT TEST_BASELINE )

message( "TEST_COMMAND ${TEST_COMMAND}")
message( "TEST_ARGSLIST ${TEST_ARGSLIST}")
message( "TEST_OUTPUT ${TEST_OUTPUT}")
message( "TEST_BASELINE ${TEST_BASELINE}")
message( "DIFF_EXECUTABLE is ${TEST_DIFF_EXECUTABLE}")

# run the test program, capture the stdout/stderr and the result var
execute_process(
  COMMAND "${TEST_COMMAND}" ${TEST_ARGSLIST}
  OUTPUT_FILE ${TEST_OUTPUT}
  ERROR_VARIABLE TEST_ERROR
  RESULT_VARIABLE TEST_RESULT
  )

# if the return value is !=0 bail out
if( TEST_RESULT )
  message( FATAL_ERROR "Failed: Test program ${TEST_PROGRAM} exited != 0.\n${TEST_ERROR}" )
endif( TEST_RESULT )

# now compare the output with the reference
execute_process(
  COMMAND ${CMAKE_COMMAND} -E compare_files ${TEST_OUTPUT} ${TEST_BASELINE}
  RESULT_VARIABLE TEST_RESULT
  )

# if return value is !=0 display something to show problems
if( TEST_RESULT )
  file(READ ${TEST_OUTPUT} ERR_TXT)
  message("#########################")
  message("OUTPUT IS ${ERR_TXT}")
  message("#########################")

  # compare the output with the reference if possible
  if (TEST_DIFF_EXECUTABLE) 
	  execute_process(
	    COMMAND ${TEST_DIFF_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/${TEST_OUTPUT}" "${TEST_BASELINE}"
      OUTPUT_VARIABLE DIFF_RESULT
	    )
    message("#########################")
    message(FATAL_ERROR "Failed: DIFF IS \n${DIFF_RESULT}")
  else (TEST_DIFF_EXECUTABLE) 
    message( FATAL_ERROR "Failed: The output of ${TEST_PROGRAM} did not match ${TEST_BASELINE}")
  endif (TEST_DIFF_EXECUTABLE) 
endif( TEST_RESULT )

# everything went fine...
message( "Passed: The output of ${TEST_PROGRAM} matches ${TEST_BASELINE}" )

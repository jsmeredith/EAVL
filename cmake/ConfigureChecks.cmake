#-----------------------------------------------------------------------------
# Include all the necessary files for macros
#-----------------------------------------------------------------------------
INCLUDE (${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFileCXX.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFiles.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckLibraryExists.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckSymbolExists.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckTypeSize.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckVariableExists.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckFortranFunctionExists.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/TestBigEndian.cmake)

#-----------------------------------------------------------------------------
# Always SET this for now IF we are on an OS X box
#-----------------------------------------------------------------------------
IF (APPLE)
  LIST(LENGTH CMAKE_OSX_ARCHITECTURES ARCH_LENGTH)
  IF(ARCH_LENGTH GREATER 1)
    set (CMAKE_OSX_ARCHITECTURES "" CACHE STRING "" FORCE)
    message(FATAL_ERROR "Building Universal Binaries on OS X is NOT supported by the EAVL project. This is"
    "due to technical reasons. The best approach would be build each architecture in separate directories"
    "and use the 'lipo' tool to combine them into a single executable or library. The 'CMAKE_OSX_ARCHITECTURES'"
    "variable has been set to a blank value which will build the default architecture for this system.")
  ENDIF()
  SET (EAVL_AC_APPLE_UNIVERSAL_BUILD 0)
ENDIF (APPLE)

#-----------------------------------------------------------------------------
# This MACRO checks IF the symbol exists in the library and IF it
# does, it appends library to the list.
#-----------------------------------------------------------------------------
SET (LINK_LIBS "")
MACRO (CHECK_LIBRARY_EXISTS_CONCAT LIBRARY SYMBOL VARIABLE)
  CHECK_LIBRARY_EXISTS ("${LIBRARY};${LINK_LIBS}" ${SYMBOL} "" ${VARIABLE})
  IF (${VARIABLE})
    SET (LINK_LIBS ${LINK_LIBS} ${LIBRARY})
  ENDIF (${VARIABLE})
ENDMACRO (CHECK_LIBRARY_EXISTS_CONCAT)

# ----------------------------------------------------------------------
# WINDOWS Hard code Values
# ----------------------------------------------------------------------

SET (WINDOWS)
IF (WIN32)
  IF (MINGW)
    SET (HAVE_MINGW 1)
    SET (WINDOWS 1) # MinGW tries to imitate Windows
  ENDIF (MINGW)
  SET (HAVE_WIN32_API 1)
  IF (NOT UNIX AND NOT CYGWIN AND NOT MINGW)
    SET (WINDOWS 1)
    IF (MSVC)
      SET (HAVE_VISUAL_STUDIO 1)
    ENDIF (MSVC)
  ENDIF (NOT UNIX AND NOT CYGWIN AND NOT MINGW)
ENDIF (WIN32)

IF (WINDOWS)
  SET (HAVE_WINDOWS 1)
  # ----------------------------------------------------------------------
  # Set the flag to indicate that the machine has window style pathname,
  # that is, "drive-letter:\" (e.g. "C:") or "drive-letter:/" (e.g. "C:/").
  # (This flag should be _unset_ for all machines, except for Windows)
  SET (HAVE_WINDOW_PATH 1)
  SET (LINK_LIBS ${LINK_LIBS} "kernel32")
ENDIF (WINDOWS)

IF (WINDOWS)
  SET (HAVE_IO_H 1)
  SET (HAVE_SETJMP_H 1)
  SET (HAVE_STDDEF_H 1)
  SET (HAVE_SYS_STAT_H 1)
  SET (HAVE_SYS_TIMEB_H 1)
  SET (HAVE_SYS_TYPES_H 1)
  SET (HAVE_WINSOCK_H 1)
  SET (HAVE_LIBM 1)
  SET (HAVE_STRDUP 1)
  SET (HAVE_SYSTEM 1)
  SET (HAVE_DIFFTIME 1)
  SET (HAVE_LONGJMP 1)
  IF (NOT MINGW)
    SET (HAVE_GETHOSTNAME 1)
  ENDIF (NOT MINGW)
  SET (HAVE_GETCONSOLESCREENBUFFERINFO 1)
  SET (HAVE_FUNCTION 1)
  SET (HAVE_TIMEZONE 1)
  SET (HAVE_GETTIMEOFDAY 1)
ENDIF (WINDOWS)

# ----------------------------------------------------------------------
# END of WINDOWS Hard code Values
# ----------------------------------------------------------------------

IF (CYGWIN)
  SET (HAVE_LSEEK64 0)
ENDIF (CYGWIN)

#-----------------------------------------------------------------------------
#  Check for the math library "m"
#-----------------------------------------------------------------------------
IF (NOT WINDOWS)
  CHECK_LIBRARY_EXISTS_CONCAT ("m" random     HAVE_LIBM)
ENDIF (NOT WINDOWS)

CHECK_LIBRARY_EXISTS_CONCAT ("ws2_32" WSAStartup     HAVE_LIBWS2_32)
CHECK_LIBRARY_EXISTS_CONCAT ("wsock32" gethostbyname HAVE_LIBWSOCK32)
CHECK_LIBRARY_EXISTS_CONCAT ("ucb"    gethostname    HAVE_LIBUCB)
CHECK_LIBRARY_EXISTS_CONCAT ("socket" connect        HAVE_LIBSOCKET)
CHECK_LIBRARY_EXISTS ("c" gethostbyname "" NOT_NEED_LIBNSL)

IF (NOT NOT_NEED_LIBNSL)
  CHECK_LIBRARY_EXISTS_CONCAT ("nsl"    gethostbyname  HAVE_LIBNSL)
ENDIF (NOT NOT_NEED_LIBNSL)


SET (USE_INCLUDES "")
IF (WINDOWS)
  SET (USE_INCLUDES ${USE_INCLUDES} "windows.h")
ENDIF (WINDOWS)

TEST_BIG_ENDIAN(EAVL_WORDS_BIGENDIAN)

#-----------------------------------------------------------------------------
# Check IF header file exists and add it to the list.
#-----------------------------------------------------------------------------
MACRO (CHECK_INCLUDE_FILE_CONCAT FILE VARIABLE)
  CHECK_INCLUDE_FILES ("${USE_INCLUDES};${FILE}" ${VARIABLE})
  IF (${VARIABLE})
    SET (USE_INCLUDES ${USE_INCLUDES} ${FILE})
  ENDIF (${VARIABLE})
ENDMACRO (CHECK_INCLUDE_FILE_CONCAT)

#-----------------------------------------------------------------------------
#  Check for the existence of certain header files
#-----------------------------------------------------------------------------
CHECK_INCLUDE_FILE_CONCAT ("globus/common.h" HAVE_GLOBUS_COMMON_H)
CHECK_INCLUDE_FILE_CONCAT ("io.h"            HAVE_IO_H)
CHECK_INCLUDE_FILE_CONCAT ("mfhdf.h"         HAVE_MFHDF_H)
CHECK_INCLUDE_FILE_CONCAT ("pdb.h"           HAVE_PDB_H)
CHECK_INCLUDE_FILE_CONCAT ("pthread.h"       HAVE_PTHREAD_H)
CHECK_INCLUDE_FILE_CONCAT ("setjmp.h"        HAVE_SETJMP_H)
CHECK_INCLUDE_FILE_CONCAT ("srbclient.h"     HAVE_SRBCLIENT_H)
CHECK_INCLUDE_FILE_CONCAT ("stddef.h"        HAVE_STDDEF_H)
CHECK_INCLUDE_FILE_CONCAT ("stdint.h"        HAVE_STDINT_H)
CHECK_INCLUDE_FILE_CONCAT ("string.h"        HAVE_STRING_H)
CHECK_INCLUDE_FILE_CONCAT ("strings.h"       HAVE_STRINGS_H)
CHECK_INCLUDE_FILE_CONCAT ("sys/ioctl.h"     HAVE_SYS_IOCTL_H)
CHECK_INCLUDE_FILE_CONCAT ("sys/proc.h"      HAVE_SYS_PROC_H)
CHECK_INCLUDE_FILE_CONCAT ("sys/resource.h"  HAVE_SYS_RESOURCE_H)
CHECK_INCLUDE_FILE_CONCAT ("sys/socket.h"    HAVE_SYS_SOCKET_H)
CHECK_INCLUDE_FILE_CONCAT ("sys/stat.h"      HAVE_SYS_STAT_H)
IF (CMAKE_SYSTEM_NAME MATCHES "OSF")
  CHECK_INCLUDE_FILE_CONCAT ("sys/sysinfo.h" HAVE_SYS_SYSINFO_H)
ELSE (CMAKE_SYSTEM_NAME MATCHES "OSF")
  SET (HAVE_SYS_SYSINFO_H "" CACHE INTERNAL "" FORCE)
ENDIF (CMAKE_SYSTEM_NAME MATCHES "OSF")
CHECK_INCLUDE_FILE_CONCAT ("sys/time.h"      HAVE_SYS_TIME_H)
CHECK_INCLUDE_FILE_CONCAT ("time.h"          HAVE_TIME_H)
CHECK_INCLUDE_FILE_CONCAT ("mach/mach_time.h" HAVE_MACH_MACH_TIME_H)
CHECK_INCLUDE_FILE_CONCAT ("sys/timeb.h"     HAVE_SYS_TIMEB_H)
CHECK_INCLUDE_FILE_CONCAT ("sys/types.h"     HAVE_SYS_TYPES_H)
CHECK_INCLUDE_FILE_CONCAT ("unistd.h"        HAVE_UNISTD_H)
CHECK_INCLUDE_FILE_CONCAT ("stdlib.h"        HAVE_STDLIB_H)
CHECK_INCLUDE_FILE_CONCAT ("memory.h"        HAVE_MEMORY_H)
CHECK_INCLUDE_FILE_CONCAT ("dlfcn.h"         HAVE_DLFCN_H)
CHECK_INCLUDE_FILE_CONCAT ("features.h"      HAVE_FEATURES_H)
CHECK_INCLUDE_FILE_CONCAT ("inttypes.h"      HAVE_INTTYPES_H)
CHECK_INCLUDE_FILE_CONCAT ("netinet/in.h"    HAVE_NETINET_IN_H)
CHECK_INCLUDE_FILE_CONCAT ("stdlib.h;stdarg.h;string.h;float.h"	STDC_HEADERS)

IF (NOT CYGWIN)
  CHECK_INCLUDE_FILE_CONCAT ("winsock2.h"      HAVE_WINSOCK_H)
ENDIF (NOT CYGWIN)

# IF the c compiler found stdint, check the C++ as well. On some systems this
# file will be found by C but not C++, only do this test IF the C++ compiler
# has been initialized (e.g. the project also includes some c++)
IF (HAVE_STDINT_H AND CMAKE_CXX_COMPILER_LOADED)
  CHECK_INCLUDE_FILE_CXX ("stdint.h" HAVE_STDINT_H_CXX)
  IF (NOT HAVE_STDINT_H_CXX)
    SET (HAVE_STDINT_H "" CACHE INTERNAL "Have includes HAVE_STDINT_H")
    SET (USE_INCLUDES ${USE_INCLUDES} "stdint.h")
  ENDIF (NOT HAVE_STDINT_H_CXX)
ENDIF (HAVE_STDINT_H AND CMAKE_CXX_COMPILER_LOADED)

#-----------------------------------------------------------------------------
#  Check the size in bytes of all the int and float types
#-----------------------------------------------------------------------------
MACRO (EAVL_CHECK_TYPE_SIZE type var)
  SET (aType ${type})
  SET (aVar  ${var})
#  MESSAGE (STATUS "Checking size of ${aType} and storing into ${aVar}")
  CHECK_TYPE_SIZE (${aType}   ${aVar})
  IF (NOT ${aVar})
    SET (${aVar} 0 CACHE INTERNAL "SizeOf for ${aType}")
#    MESSAGE (STATUS "Size of ${aType} was NOT Found")
  ENDIF (NOT ${aVar})
ENDMACRO (EAVL_CHECK_TYPE_SIZE)


EAVL_CHECK_TYPE_SIZE (char           SIZEOF_CHAR)
EAVL_CHECK_TYPE_SIZE (short          SIZEOF_SHORT)
EAVL_CHECK_TYPE_SIZE (int            SIZEOF_INT)
EAVL_CHECK_TYPE_SIZE (unsigned       SIZEOF_UNSIGNED)
IF (NOT APPLE)
  EAVL_CHECK_TYPE_SIZE (long         SIZEOF_LONG)
ENDIF (NOT APPLE)
EAVL_CHECK_TYPE_SIZE ("long long"    SIZEOF_LONG_LONG)
EAVL_CHECK_TYPE_SIZE (__int64        SIZEOF___INT64)
EAVL_CHECK_TYPE_SIZE ("void *"       SIZEOF_VOID_P)
IF (NOT SIZEOF___INT64)
  SET (SIZEOF___INT64 0)
ENDIF (NOT SIZEOF___INT64)

EAVL_CHECK_TYPE_SIZE (float          SIZEOF_FLOAT)
EAVL_CHECK_TYPE_SIZE (double         SIZEOF_DOUBLE)
EAVL_CHECK_TYPE_SIZE ("long double"  SIZEOF_LONG_DOUBLE)
EAVL_CHECK_TYPE_SIZE (int8_t         SIZEOF_INT8_T)
EAVL_CHECK_TYPE_SIZE (uint8_t        SIZEOF_UINT8_T)
EAVL_CHECK_TYPE_SIZE (int_least8_t   SIZEOF_INT_LEAST8_T)
EAVL_CHECK_TYPE_SIZE (uint_least8_t  SIZEOF_UINT_LEAST8_T)
EAVL_CHECK_TYPE_SIZE (int_fast8_t    SIZEOF_INT_FAST8_T)
EAVL_CHECK_TYPE_SIZE (uint_fast8_t   SIZEOF_UINT_FAST8_T)
EAVL_CHECK_TYPE_SIZE (int16_t        SIZEOF_INT16_T)
EAVL_CHECK_TYPE_SIZE (uint16_t       SIZEOF_UINT16_T)
EAVL_CHECK_TYPE_SIZE (int_least16_t  SIZEOF_INT_LEAST16_T)
EAVL_CHECK_TYPE_SIZE (uint_least16_t SIZEOF_UINT_LEAST16_T)
EAVL_CHECK_TYPE_SIZE (int_fast16_t   SIZEOF_INT_FAST16_T)
EAVL_CHECK_TYPE_SIZE (uint_fast16_t  SIZEOF_UINT_FAST16_T)
EAVL_CHECK_TYPE_SIZE (int32_t        SIZEOF_INT32_T)
EAVL_CHECK_TYPE_SIZE (uint32_t       SIZEOF_UINT32_T)
EAVL_CHECK_TYPE_SIZE (int_least32_t  SIZEOF_INT_LEAST32_T)
EAVL_CHECK_TYPE_SIZE (uint_least32_t SIZEOF_UINT_LEAST32_T)
EAVL_CHECK_TYPE_SIZE (int_fast32_t   SIZEOF_INT_FAST32_T)
EAVL_CHECK_TYPE_SIZE (uint_fast32_t  SIZEOF_UINT_FAST32_T)
EAVL_CHECK_TYPE_SIZE (int64_t        SIZEOF_INT64_T)
EAVL_CHECK_TYPE_SIZE (uint64_t       SIZEOF_UINT64_T)
EAVL_CHECK_TYPE_SIZE (int_least64_t  SIZEOF_INT_LEAST64_T)
EAVL_CHECK_TYPE_SIZE (uint_least64_t SIZEOF_UINT_LEAST64_T)
EAVL_CHECK_TYPE_SIZE (int_fast64_t   SIZEOF_INT_FAST64_T)
EAVL_CHECK_TYPE_SIZE (uint_fast64_t  SIZEOF_UINT_FAST64_T)
IF (NOT APPLE)
  EAVL_CHECK_TYPE_SIZE (size_t       SIZEOF_SIZE_T)
  EAVL_CHECK_TYPE_SIZE (ssize_t      SIZEOF_SSIZE_T)
  IF (NOT SIZEOF_SSIZE_T)
    SET (SIZEOF_SSIZE_T 0)
  ENDIF (NOT SIZEOF_SSIZE_T)
ENDIF (NOT APPLE)
EAVL_CHECK_TYPE_SIZE (off_t          SIZEOF_OFF_T)
EAVL_CHECK_TYPE_SIZE (off64_t        SIZEOF_OFF64_T)
IF (NOT SIZEOF_OFF64_T)
  SET (SIZEOF_OFF64_T 0)
ENDIF (NOT SIZEOF_OFF64_T)


# For other tests to use the same libraries
SET (CMAKE_REQUIRED_LIBRARIES ${LINK_LIBS})

#-----------------------------------------------------------------------------
# Check for some functions that are used
#
CHECK_FUNCTION_EXISTS (alarm             HAVE_ALARM)
CHECK_FUNCTION_EXISTS (fork              HAVE_FORK)
CHECK_FUNCTION_EXISTS (frexpf            HAVE_FREXPF)
CHECK_FUNCTION_EXISTS (frexpl            HAVE_FREXPL)

CHECK_FUNCTION_EXISTS (gethostname       HAVE_GETHOSTNAME)
CHECK_FUNCTION_EXISTS (getpwuid          HAVE_GETPWUID)
CHECK_FUNCTION_EXISTS (getrusage         HAVE_GETRUSAGE)
CHECK_FUNCTION_EXISTS (lstat             HAVE_LSTAT)

CHECK_FUNCTION_EXISTS (rand_r            HAVE_RAND_R)
CHECK_FUNCTION_EXISTS (random            HAVE_RANDOM)
CHECK_FUNCTION_EXISTS (setsysinfo        HAVE_SETSYSINFO)

CHECK_FUNCTION_EXISTS (signal            HAVE_SIGNAL)
CHECK_FUNCTION_EXISTS (longjmp           HAVE_LONGJMP)
CHECK_FUNCTION_EXISTS (setjmp            HAVE_SETJMP)
CHECK_FUNCTION_EXISTS (siglongjmp        HAVE_SIGLONGJMP)
CHECK_FUNCTION_EXISTS (sigsetjmp         HAVE_SIGSETJMP)
CHECK_FUNCTION_EXISTS (sigaction         HAVE_SIGACTION)
CHECK_FUNCTION_EXISTS (sigprocmask       HAVE_SIGPROCMASK)

CHECK_FUNCTION_EXISTS (snprintf          HAVE_SNPRINTF)
CHECK_FUNCTION_EXISTS (srandom           HAVE_SRANDOM)
CHECK_FUNCTION_EXISTS (strdup            HAVE_STRDUP)
CHECK_FUNCTION_EXISTS (symlink           HAVE_SYMLINK)
CHECK_FUNCTION_EXISTS (system            HAVE_SYSTEM)

CHECK_FUNCTION_EXISTS (tmpfile           HAVE_TMPFILE)
CHECK_FUNCTION_EXISTS (vasprintf         HAVE_VASPRINTF)
CHECK_FUNCTION_EXISTS (waitpid           HAVE_WAITPID)

CHECK_FUNCTION_EXISTS (vsnprintf         HAVE_VSNPRINTF)
CHECK_FUNCTION_EXISTS (ioctl             HAVE_IOCTL)
#CHECK_FUNCTION_EXISTS (gettimeofday      HAVE_GETTIMEOFDAY)
CHECK_FUNCTION_EXISTS (difftime          HAVE_DIFFTIME)
CHECK_FUNCTION_EXISTS (fseeko            HAVE_FSEEKO)
CHECK_FUNCTION_EXISTS (ftello            HAVE_FTELLO)
CHECK_FUNCTION_EXISTS (fseeko64          HAVE_FSEEKO64)
CHECK_FUNCTION_EXISTS (ftello64          HAVE_FTELLO64)
CHECK_FUNCTION_EXISTS (fstat64           HAVE_FSTAT64)
CHECK_FUNCTION_EXISTS (stat64            HAVE_STAT64)

#-----------------------------------------------------------------------------
# sigsetjmp is special; may actually be a macro
IF (NOT HAVE_SIGSETJMP)
  IF (HAVE_SETJMP_H)
    CHECK_SYMBOL_EXISTS (sigsetjmp "setjmp.h" HAVE_MACRO_SIGSETJMP)
    IF (HAVE_MACRO_SIGSETJMP)
      SET (HAVE_SIGSETJMP 1)
    ENDIF (HAVE_MACRO_SIGSETJMP)
  ENDIF (HAVE_SETJMP_H)
ENDIF (NOT HAVE_SIGSETJMP)


# Check for Symbols
CHECK_SYMBOL_EXISTS (tzname "time.h" HAVE_DECL_TZNAME)

#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
IF (NOT WINDOWS)
  CHECK_SYMBOL_EXISTS (TIOCGWINSZ "sys/ioctl.h" HAVE_TIOCGWINSZ)
  CHECK_SYMBOL_EXISTS (TIOCGETD   "sys/ioctl.h" HAVE_TIOCGETD)
ENDIF (NOT WINDOWS)


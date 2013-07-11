#-------------------------------------------------------------------------------
MACRO (INIT_GLOBAL_LIST name value)
  SET (${name} ${value} CACHE INTERNAL "Used to pass variables between directories" FORCE)
ENDMACRO (SET_GLOBAL_VARIABLE)
#-------------------------------------------------------------------------------
MACRO (ADD_GLOBAL_LIST name value)
  SET (${name} "${${name}};${value}" CACHE INTERNAL "Used to pass variables between directories" FORCE)
ENDMACRO (ADD_GLOBAL_LIST)
#-------------------------------------------------------------------------------

SET(EAVL_FONTS_SRCS
  eavlBitmapFontFactory.cpp
  Liberation2Mono.cpp
  Liberation2Sans.cpp
  Liberation2Serif.cpp
  eavlBitmapFont.cpp
)

add_library(eavl_fonts
  ${EAVL_FONTS_SRCS}
)
ADD_GLOBAL_LIST(EAVL_EXPORTED_LIBS eavl_fonts)

SET(EAVL_FONTS_SRCS
  eavlBitmapFont.cpp
  eavlBitmapFontFactory.cpp
  Liberation2Mono.cpp
  Liberation2Sans.cpp
  Liberation2Serif.cpp
)

add_library(eavl_fonts
  ${EAVL_FONTS_SRCS}
)
ADD_GLOBAL_LIST(EAVL_EXPORTED_LIBS eavl_fonts)

// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_BITMAP_FONT_FACTORY_H
#define EAVL_BITMAP_FONT_FACTORY_H

#include "STL.h"

class eavlBitmapFont;

// ****************************************************************************
// Class:  eavlBitmapFontFactory
//
// Purpose:
///   Create fonts as needed.
//
// Programmer:  Jeremy Meredith
// Creation:    January  8, 2013
//
// Modifications:
// ****************************************************************************
class eavlBitmapFontFactory
{
  public:
    static eavlBitmapFont *GetDefaultFont();
    static eavlBitmapFont *GetFont(const std::string &name);
};


#endif

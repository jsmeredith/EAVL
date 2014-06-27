// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlBitmapFontFactory.h"
#include "eavlBitmapFont.h"

#include "Liberation2Mono.h"
#include "Liberation2Sans.h"
#include "Liberation2Serif.h"

static eavlBitmapFont *defaultFont = NULL;

static map<string,eavlBitmapFont*> allFonts;

eavlBitmapFont*
eavlBitmapFontFactory::GetDefaultFont()
{
    if (!defaultFont)
    {
        defaultFont = CreateLiberation2SansFont();
        allFonts[defaultFont->name] = defaultFont;
    }
    return defaultFont;
}

eavlBitmapFont*
eavlBitmapFontFactory::GetFont(const std::string &name)
{
    if (allFonts.count(name))
        return allFonts[name];

    eavlBitmapFont *font = NULL;
    ///\todo: note: the names here must match exactly what
    /// is in font->name.   we should improve that so it
    /// can be done automatically (though that would require
    /// initializing all fonts at startup....)  also, we
    /// are currently ignoring style....
    if (name == "Liberation Mono")
        font = CreateLiberation2MonoFont();
    if (name == "Liberation Sans")
        font = CreateLiberation2SansFont();
    if (name == "Liberation Serif")
        font = CreateLiberation2SerifFont();

    if (!font)
        return GetDefaultFont();

    allFonts[name] = font;
    return font;
}

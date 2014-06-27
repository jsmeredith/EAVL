// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_ANNOTATION_H
#define EAVL_ANNOTATION_H

class eavlWindow;
#include "eavlView.h"

// ****************************************************************************
// Class:  eavlAnnotation
//
// Purpose:
///   Base class for all GL annotations.
//
// Programmer:  Jeremy Meredith
// Creation:    January 11, 2013
//
// Modifications:
// ****************************************************************************

///\todo: Rename to "eavlGLDrawable" or something like that....
/// Also implies the various annotations should be renamed to include "GL"?
class eavlAnnotation
{
  protected:
    eavlWindow *win;
  public:
    eavlAnnotation(eavlWindow *w)
        : win(w)
    {
    }


    virtual void Render(eavlView &view) = 0;

  protected:
};

#endif

// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WORLD_ANNOTATOR_PS_H
#define EAVL_WORLD_ANNOTATOR_PS_H

#include "eavlWorldAnnotator.h"

class eavlWorldAnnotatorPS : public eavlWorldAnnotator
{
  protected:

  public:
    eavlWorldAnnotatorPS() : eavlWorldAnnotator()
    {
    }
    virtual void AddLine(float x0, float y0, float z0,
                         float x1, float y1, float z1,
                         float linewidth,
                         eavlColor c,
                         bool infront)
    {
    }
    virtual void AddText(float ox, float oy, float oz,
                         float rx, float ry, float rz,
                         float ux, float uy, float uz,
                         float scale,
                         float anchorx, float anchory,
                         eavlColor color,
                         string text)
    {
    }
};

#endif


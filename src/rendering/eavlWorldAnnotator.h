// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WORLD_ANNOTATOR_H
#define EAVL_WORLD_ANNOTATOR_H

class eavlWorldAnnotator
{
  public:
    eavlWorldAnnotator()
    {
    }
    virtual void AddLine(float x0, float y0, float z0,
                         float x1, float y1, float z1,
                         float linewidth,
                         eavlColor c, bool infront=false) = 0;
};

#endif


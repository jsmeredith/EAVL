// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WORLD_ANNOTATOR_GL_H
#define EAVL_WORLD_ANNOTATOR_GL_H

#include "eavlWorldAnnotator.h"

class eavlWorldAnnotatorGL : public eavlWorldAnnotator
{
  public:
    eavlWorldAnnotatorGL() : eavlWorldAnnotator()
    {
    }
    virtual void AddLine(float x0, float y0, float z0,
                         float x1, float y1, float z1,
                         float linewidth,
                         eavlColor c,
                         bool infront)
    {
        if (infront)
            glDepthRange(-.0001,.9999);

        glDisable(GL_LIGHTING);

        glColor3fv(c.c);

        glLineWidth(linewidth);

        glBegin(GL_LINES);
        glVertex3f(x0,y0,z0);
        glVertex3f(x1,y1,z1);
        glEnd();

        if (infront)
            glDepthRange(0,1);

    }
};

#endif


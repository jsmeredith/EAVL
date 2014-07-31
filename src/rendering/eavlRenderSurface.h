#ifndef EAVL_RENDER_SURFACE_H
#define EAVL_RENDER_SURFACE_H

#include "eavlColor.h"

class eavlRenderSurface
{
  protected:
  public:
    eavlRenderSurface()
    {
    }
    virtual void Initialize() = 0;
    virtual void Resize(int w, int h) = 0;
    virtual void Activate() = 0;
    virtual void Finish() = 0;
    //virtual unsigned char *GetRGBA()

    virtual void AddLine(float x0, float y0,
                         float x1, float y1,
                         float linewidth,
                         eavlColor c) = 0;
    virtual void AddRectangle(float x, float y, 
                              float w, float h,
                              eavlColor c) = 0;
    virtual void AddColorBar(float x, float y, 
                             float w, float h,
                             string ctname,
                             bool horizontal) = 0;
};

#endif

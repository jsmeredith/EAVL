// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_RENDER_SURFACE_PX_H
#define EAVL_RENDER_SURFACE_PX_H

#include "eavlRenderSurface.h"
#include "eavlException.h"

#include <climits>

class eavlRenderSurfacePX : public eavlRenderSurface
{
  protected:
    vector<unsigned char> rgba;
    vector<float> zbuff;
    int width, height;
  public:
    eavlRenderSurfacePX() : eavlRenderSurface(), width(0),height(0)
    {
    }
    virtual ~eavlRenderSurfacePX()
    {
    }
    virtual void Initialize()
    {
    }
    virtual void Resize(int w, int h)
    {
        width = w;
        height = h;
        rgba.resize(w*h*4);
        zbuff.resize(w*h);
    }
    virtual void Activate()
    {
    }
    virtual void Finish()
    {
    }
    virtual void Clear(eavlColor bg)
    {
    }

    virtual void SetViewToWorldSpace(eavlView &v, bool clip)
    {
    }
    virtual void SetViewToScreenSpace(eavlView &v, bool clip)
    {
    }
    void AddViewportClipPath(eavlView &v)
    {
    }

    virtual void AddRectangle(float x, float y, 
                              float w, float h,
                              eavlColor c)
    {
    }
    virtual void AddLine(float x0, float y0,
                         float x1, float y1,
                         float linewidth,
                         eavlColor c)
    {
    }
    virtual void AddColorBar(float x, float y, 
                             float w, float h,
                             const eavlColorTable &ct,
                             bool horizontal)
    {
    }

    virtual void AddText(float x, float y,
                         float scale,
                         float angle,
                         float windowaspect,
                         float anchorx, float anchory,
                         eavlColor color,
                         string text) ///<\todo: better way to get view here!
    {
    }
    virtual void PasteScenePixels(int w, int h,
                                  unsigned char *newrgba,
                                  float *newdepth)
    {
        for (int i=0; i<w*h*4; ++i)
        {
            rgba[i] = newrgba[i];
        }
    }
    virtual void SaveAs(string fn, FileType ft)
    {
        if (ft != PNM)
        {
            THROW(eavlException, "Can only save PX images as PNM");
        }

        int w = width, h = height;

        ofstream out(fn.c_str());
        out<<"P6"<<endl<<w<<" "<<h<<endl<<255<<endl;
        for(int i = h-1; i >= 0; i--)
        {
            for(int j = 0; j < w; j++)
            {
                const byte *tuple = &(rgba[i*w*4 + j*4]);
                out<<tuple[0]<<tuple[1]<<tuple[2];
            }
        }
        out.close();

    }
};

#endif

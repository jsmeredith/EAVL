// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WINDOW_H
#define EAVL_WINDOW_H

#include "eavl.h"
#include "eavlView.h"
#include "eavlRenderer.h"
#include "eavlColorTable.h"
#include "eavlPlot.h"
#include "eavlTexture.h"
#include "eavlRenderSurface.h"

class eavlScene;
// ****************************************************************************
// Class:  eavlWindow
//
// Purpose:
///   Encapsulate an output window (e.g. 3D).
//
// Programmer:  Jeremy Meredith
// Creation:    January 23, 2013
//
// Modifications:
// ****************************************************************************
class eavlWindow
{
  protected:
    eavlColor bg;
    eavlScene *scene;
    std::map<std::string,eavlTexture*> textures;

  public: /// todo: hack, should not be public
    eavlRenderSurface *surface;
    eavlView view;

  public:
    eavlWindow(eavlColor bgcolor, eavlScene *s = NULL) : bg(bgcolor), scene(s), surface(NULL)
    {
    }

    /*
    virtual void ResetViewForCurrentExtents() { }
    */

    void Initialize()
    {
        if (surface)
            surface->Initialize();
    }
    void Resize(int w, int h)
    {
        if (surface)
            surface->Resize(w,h);

        view.w = w;
        view.h = h;
    }
    void Paint()
    {
        if (surface)
            surface->Activate();

        view.SetupMatrices();
        glClearColor(bg.c[0], bg.c[1], bg.c[2], 1.0); ///< c[3] instead of 1.0?
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

        // render the plots and annotations
        Render();

        glFinish();
    }

    /*
    unsigned char *GetRGBABuffer()
    {
    }
    *GetZBuffer()
    {
    }
    void PasteRGBABuffer(unsigned char *)
    {
    }
    */

    virtual void Render() = 0;

    eavlTexture *GetTexture(const std::string &s)
    {
        return textures[s];
    }
    void SetTexture(const std::string &s, eavlTexture *tex)
    {
        textures[s] = tex;
    }

  protected:
  public:
};

#endif


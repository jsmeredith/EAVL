// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WINDOW_H
#define EAVL_WINDOW_H

#include "eavl.h"
#include "eavlView.h"
#include "eavlRenderer.h"
#include "eavlColorTable.h"
#include "eavlPlot.h"
#include "eavlTexture.h"

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
    eavlScene *scene;
    std::map<std::string,eavlTexture*> textures;

  public: /// todo: hack, should not be public
    eavlView view;

  public:
    eavlWindow(eavlScene *s = NULL) : scene(s)
    {
    }

    /*
    virtual void ResetViewForCurrentExtents() { }
    */

    virtual void Initialize() { }
    virtual void Resize(int w, int h)
    {
        view.w = w;
        view.h = h;
    }
    virtual void Paint() = 0;

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


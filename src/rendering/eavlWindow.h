// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WINDOW_H
#define EAVL_WINDOW_H

#include "eavl.h"
#include "eavlView.h"
#include "eavlColor.h"
#include "eavlRenderSurface.h"
#include "eavlSceneRenderer.h"
#include "eavlAnnotation.h"
#include "eavlWorldAnnotator.h"

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
    eavlSceneRenderer *renderer;

    std::vector<eavlAnnotation*> annotations;

  public: /// todo: hack, should not be public
    ///\todo: no longer allow a NULL surface!
    eavlRenderSurface *surface;
    eavlWorldAnnotator *worldannotator;
    eavlView view;

  public:
    eavlWindow(eavlColor bgcolor, eavlRenderSurface *surf,
               eavlScene *s, eavlSceneRenderer *r,
               eavlWorldAnnotator *w)
        : bg(bgcolor), scene(s), renderer(r), surface(surf), worldannotator(w)
    {
    }
    virtual ~eavlWindow()
    {
        //delete scene;
    }

    void SetSceneRenderer(eavlSceneRenderer *sr)
    {
        if (renderer)
            delete renderer;
        renderer = sr;
    }
    eavlSceneRenderer *GetSceneRenderer()
    {
        return renderer;
    }
    void ClearAnnotations()
    {
        annotations.clear();
    }

    void AddAnnotation(eavlAnnotation *ann)
    {
        annotations.push_back(ann);
    }

    /*
    virtual void ResetViewForCurrentExtents() { }
    */

    void Initialize()
    {
        ///\todo: we want to make sure initialize called before resize/paint?
        surface->Initialize();
    }
    void Resize(int w, int h)
    {
        surface->Resize(w,h);

        view.w = w;
        view.h = h;
    }
    void Paint()
    {
        surface->Activate();
        surface->Clear(bg);

        view.SetupMatrices();

        // render the plots and annotations
        Render();

        for (unsigned int i=0; i<annotations.size(); ++i)
            annotations[i]->Render(view);

        surface->Finish();
    }

    virtual void Render() = 0;
};

#endif


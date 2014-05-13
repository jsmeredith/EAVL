// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WINDOW_H
#define EAVL_WINDOW_H

#include "eavl.h"
#include "eavlView.h"
#include "eavlColor.h"
#include "eavlTexture.h"
#include "eavlRenderSurface.h"
#include "eavlAnnotation.h"

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

    std::vector<eavlAnnotation*> annotations;

  public: /// todo: hack, should not be public
    eavlRenderSurface *surface;
    eavlView view;

  public:
    eavlWindow(eavlColor bgcolor, eavlRenderSurface *surf,
               eavlScene *s = NULL) : bg(bgcolor), scene(s), surface(surf)
    {
    }
    virtual ~eavlWindow()
    {
        //delete scene;
        for (std::map<std::string,eavlTexture*>::iterator i = textures.begin();
             i != textures.end() ; ++i)
            delete i->second;
        textures.clear();
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

        for (unsigned int i=0; i<annotations.size(); ++i)
            annotations[i]->Render(view);

        glFinish();

        if (surface)
            surface->Finish();
    }

    virtual void Render() = 0;

    eavlTexture *GetTexture(const std::string &s)
    {
        return textures[s];
    }
    void SetTexture(const std::string &s, eavlTexture *tex)
    {
        textures[s] = tex;
    }

    void SaveWindowAsPNM(const std::string &fn)
    {
        if (surface)
            surface->Activate();

        int w = view.w, h = view.h;
        vector<byte> rgba(w*h*4);
        glReadPixels(0,0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, &rgba[0]);

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


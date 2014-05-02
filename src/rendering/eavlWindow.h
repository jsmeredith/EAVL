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
    inline void AddWindowAnnotation(const std::string &str,
                             double ox, double oy,
                             double ah, double av,
                             double fontscale = 0.05,
                             double angle = 0.0);

    inline void AddViewportAnnotation(const std::string &str,
                                      double vx, double vy,
                                      double dx, double dy,
                                      double ah, double av,
                                      double fontscale = 0.05,
                                      double angle = 0.0);

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

#include "eavlTextAnnotation.h"


// Some quick usage examples while this is in progress:

// First, note that window (and I've decided viewport)
// coordinates are -1 to +1 (l to r, b to t).
// Text anchors are (currently) 0 to 1 (l to r, b to t).

// Title text: centered at the top of the window
//window->AddWindowAnnotation("title", 0,1, .5,1, 0.08);
// So window coordinates are 0,1 (top hcenter)
// Anchor is .5 1 (top hcenter)

// X axis: centered below the x axis labels
//window->AddViewportAnnotation("xaxis", 0,-1, 0,-.1, .5,1, 0.05);
// viewport anchor is 0,-1 (start at hcenter bottom of viewport)
// then move 0,-.1 (down about two font heights)
// and anchor at .5,1 (hcenter top) to have the text go down from there

// Y axis: vcentered to the left of the y axis labels
//window->AddViewportAnnotation("yaxis", -1,0, -.1,0, .5,0, 0.05, 90);
// viewport anchor is -1,0 (start at left vcenter of viewport)
// then a WINDOW, ASPECT-INDEPENDENT offset of -.1,0
//    (which should get it west of much yaxis tick labels)
//    (this is the only set of coordinates I've found that
//    kind of needs to have the x value modulated by the window aspect ratio)
// then anchor the text against its (after rotation) right vcenter edge, BUT
//   note that the text position values used for an anchor is BEFORE
//   rotation; so we actually use the bottom hcenter as the anchor (.5 0)
// and of course use a normal font size (0.05) and 90 degree (ccw) rotation

// two lines in the bottom lower-right of viewport:
//window->AddViewportAnnotation("line2", 1,-1, 0,0,   1,0, 0.05);
//window->AddViewportAnnotation("line1", 1,-1, 0,.05, 1,0, 0.05);
// These both start with a (1,-1) (bottom right) viewport anchor
// For clarity, we add them from the bottom-up.
// The first has a 0,0 offset and a lower-right (1,0) anchor.
// The second has a 0,0.05 (one text height up) offset and the same (1,0) anchor.

inline void eavlWindow::AddWindowAnnotation(const std::string &str,
                                     double ox, double oy,
                                     double ah, double av,
                                     double fontscale,
                                     double angle)
{
    eavlColor fg = bg.RawBrightness() < 0.5 ? eavlColor::white : eavlColor::black;
    eavlScreenTextAnnotation *t = 
        new eavlScreenTextAnnotation(this, str, fg,
                                     fontscale,
                                     ox, oy, angle);
    t->SetAnchor(ah, av);
    annotations.push_back(t);
}

inline void eavlWindow::AddViewportAnnotation(const std::string &str,
                                     double vx, double vy,
                                     double dx, double dy,
                                     double ah, double av,
                                     double fontscale,
                                     double angle)
{
    double vl=view.vl, vr=view.vr, vb=view.vb, vt=view.vt;
    if (view.viewtype == eavlView::EAVL_VIEW_2D)
        view.GetRealViewport(vl,vr,vb,vt);

    double ox = dx/view.windowaspect + (vl+vr)/2. + vx * (vr-vl)/2.;
    double oy = dy + (vb+vt)/2. + vy * (vt-vb)/2.;

    eavlColor fg = bg.RawBrightness() < 0.5 ? eavlColor::white : eavlColor::black;
    eavlScreenTextAnnotation *t = 
        new eavlScreenTextAnnotation(this, str, fg,
                                     fontscale,
                                     ox, oy, angle);
    t->SetAnchor(ah, av);
    annotations.push_back(t);
}

#endif


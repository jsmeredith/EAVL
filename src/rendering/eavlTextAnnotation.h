// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TEXT_ANNOTATION_H
#define EAVL_TEXT_ANNOTATION_H

#include <eavlMatrix4x4.h>
#include <eavlAnnotation.h>

#ifdef _WIN32
 #include "GL/glext.h"
#endif

// ****************************************************************************
// Class:  eavlTextAnnotation
//
// Purpose:
///   Allows 2D or 3D text.
//
// Programmer:  Jeremy Meredith
// Creation:    January  9, 2013
//
// Modifications:
// ****************************************************************************
class eavlTextAnnotation : public eavlAnnotation
{
  public:
    enum HorizontalAlignment
    {
        Left,
        HCenter,
        Right
    };
    enum VerticalAlignment
    {
        Bottom,
        VCenter,
        Top
    };

  protected:
    string text;
    eavlColor color;
    double  scale;
    double anchorx, anchory;

  public:
    eavlTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, double s)
        : eavlAnnotation(w), text(txt), color(c), scale(s)
    {
        // default anchor: bottom-left
        anchorx = -1;
        anchory = -1;
    }
    virtual ~eavlTextAnnotation()
    {
    }
    void SetText(const string &txt)
    {
        text = txt;
    }
    void SetRawAnchor(double h, double v)
    {
        anchorx = h;
        anchory = v;
    }
    void SetAlignment(HorizontalAlignment h, VerticalAlignment v)
    {
        switch (h)
        {
          case Left:    anchorx = -1.0; break;
          case HCenter: anchorx =  0.0; break;
          case Right:   anchorx = +1.0; break;
        }

        // For vertical alignment, "center" is generally the center
        // of only the above-baseline contents of the font, so we
        // use a value slightly off of zero for VCenter.
        // (We don't use an offset value instead of -1.0 for the 
        // bottom value, because generally we want a true minimum
        // extent, e.g. to have text sitting at the bottom of a
        // window, and in that case, we need to keep all the text,
        // including parts that descend below the baseline, above
        // the bottom of the window.
        switch (v)
        {
          case Bottom:  anchory = -1.0;  break;
          case VCenter: anchory = -0.06; break;
          case Top:     anchory = +1.0;  break;
        }
    }
    void SetScale(double s)
    {
        scale = s;
    }
};

// ****************************************************************************
// Class:  eavlScreenTextAnnotation
//
// Purpose:
///   Text location and height are in normalized screen space coordinates.
///   At the default angle (0.0), text is oriented upright.
//
// Programmer:  Jeremy Meredith
// Creation:    January 10, 2013
//
// Modifications:
// ****************************************************************************
class eavlScreenTextAnnotation : public eavlTextAnnotation
{
  protected:
    double x,y;
    double angle;
  public:
    eavlScreenTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, double s,
                             double ox, double oy, double angleDeg = 0.)
        : eavlTextAnnotation(w,txt,c,s)
    {
        x = ox;
        y = oy;
        angle = angleDeg;
    }
    void SetPosition(double ox, double oy)
    {
        x = ox;
        y = oy;
    }
    virtual void Render(eavlView &view)
    {
        win->SetupForScreenSpace();
        win->surface->AddText(x,y,
                              scale,
                              angle,
                              view.windowaspect,
                              anchorx, anchory,
                              color, text);
    }
};

// ****************************************************************************
// Class:  eavlWorldTextAnnotation
//
// Purpose:
///   Text location, orientation, and size are in world space coordinates.
//
// Programmer:  Jeremy Meredith
// Creation:    January 10, 2013
//
// Modifications:
// ****************************************************************************
class eavlWorldTextAnnotation : public eavlTextAnnotation
{
  protected:
    // new way: store locations
    eavlPoint3 origin;
    eavlVector3 normal;
    eavlVector3 up;
    // old way: create matrix at construction
    eavlMatrix4x4 mtx;
  public:
    eavlWorldTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, double s,
                            double ox, double oy, double oz,
                            double nx, double ny, double nz,
                            double ux, double uy, double uz)
        : eavlTextAnnotation(w,txt,c,s),
          origin(ox,oy,oz),
          normal(nx,ny,nz),
          up(ux,uy,uz)
    {
        up = up.normalized();
        normal = normal.normalized();
        // old way
        mtx.CreateRBT(eavlPoint3(ox,oy,oz),
                      eavlPoint3(ox,oy,oz) - eavlVector3(nx,ny,nz),
                      eavlVector3(ux,uy,uz));
    }
    virtual void Render(eavlView &view)
    {
        win->SetupForWorldSpace();

        eavlVector3 right = (up % normal).normalized();
        win->worldannotator->AddText(origin.x,origin.y,origin.z,
                                     right.x, right.y, right.z,
                                     up.x,    up.y,    up.z,
                                     scale,
                                     anchorx,anchory,
                                     color,text);
    }
};

// ****************************************************************************
// Class:  eavlBillboardTextAnnotation
//
// Purpose:
///   Text location origin is in world space, but the text is rotated so it
///   is always facing towards the user and always at the same orientation
///   (e.g. upright if angle==0).
///   Height can either be in screen space height (so it doesn't change
///   apparent size as the view moves), or in world space height (so
///   it gets bigger and smaller based on distance to the viewer).
//
// Programmer:  Jeremy Meredith
// Creation:    January 10, 2013
//
// Modifications:
// ****************************************************************************
class eavlBillboardTextAnnotation : public eavlTextAnnotation
{
  protected:
    double x,y,z;
    bool fixed2Dscale;
    double angle;
  public:
    eavlBillboardTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, double s,
                                double ox, double oy, double oz,
                                bool scaleIsScreenSpace,
                                double angleDeg = 0.)
        : eavlTextAnnotation(w,txt,c,s)
    {
        x = ox;
        y = oy;
        z = oz;
        angle = angleDeg;
        fixed2Dscale = scaleIsScreenSpace;
    }
    void SetPosition(double ox, double oy, double oz)
    {
        x = ox;
        y = oy;
        z = oz;
    }
    virtual void Render(eavlView &view)
    {
        if (fixed2Dscale)
        {
            // use the world view matrices to find out screen space points
            view.SetupMatrices();
            eavlPoint3 p = view.P * view.V * eavlPoint3(x,y,z);

            // and now everything in screen space, but still clip against
            // world viewport
            win->SetupForScreenSpace(true);

            eavlMatrix4x4 T;
            T.CreateTranslate(p.x, p.y, -p.z);

            eavlMatrix4x4 SW;
            SW.CreateScale(1./view.windowaspect, 1, 1);

            eavlMatrix4x4 SV;
            SV.CreateIdentity();
            //if (view.viewtype == eavlView::EAVL_VIEW_2D)
            {
                double vl, vr, vt, vb;
                view.GetRealViewport(vl,vr,vb,vt);
                double xs = (vr-vl);
                double ys = (vt-vb);
                SV.CreateScale(2./xs, 2./ys, 1);
            }

            eavlMatrix4x4 R;
            R.CreateRotateZ(angle * M_PI / 180.);

            eavlMatrix4x4 M = T * SW * SV * R;

            eavlPoint3 origin(0,0,0);
            eavlVector3 right(1,0,0);
            eavlVector3 up(0,1,0);

            origin = M * origin;
            right = M * right;
            up = M * up;
            win->worldannotator->AddText(origin.x,origin.y,origin.z,
                                         right.x, right.y, right.z,
                                         up.x,    up.y,    up.z,
                                         scale,
                                         anchorx,anchory,
                                         color,text);
        }
        else
        {
            win->SetupForWorldSpace(true);

            eavlMatrix4x4 W;
            if (view.viewtype == eavlView::EAVL_VIEW_2D)
            {
                W.CreateRBT(eavlPoint3(x,y,z),
                            eavlPoint3(x,y,z) - eavlVector3(0,0,1),
                            eavlVector3(0,1,0));
            }
            else
            {
                W.CreateRBT(eavlPoint3(x,y,z),
                            eavlPoint3(x,y,z) - (view.view3d.from-view.view3d.at),
                            view.view3d.up);
            }

            eavlMatrix4x4 S;
            S.CreateIdentity();
            if (view.viewtype == eavlView::EAVL_VIEW_2D)
            {
                S.CreateScale(1. / view.view2d.xscale, 1, 1);
            }

            eavlMatrix4x4 R;
            R.CreateRotateZ(angle * M_PI / 180.);

            eavlMatrix4x4 M = W * S * R;

            eavlPoint3 origin(0,0,0);
            eavlVector3 right(1,0,0);
            eavlVector3 up(0,1,0);

            origin = M * origin;
            right = M * right;
            up = M * up;
            win->worldannotator->AddText(origin.x,origin.y,origin.z,
                                         right.x, right.y, right.z,
                                         up.x,    up.y,    up.z,
                                         scale,
                                         anchorx,anchory,
                                         color,text);
        }
    }
};

// ****************************************************************************
// Class:  eavlViewportAnchoredScreenTextAnnotation
//
// Purpose:
///   Screen text is anchored to a normalized viewport location instead of
///   window location, then offset in (aspect-independent) window
///   coordinates.  The aspect-independence means that (like font size
///   making sense no matter the window aspect ratio and text rotation)
///   the x and y offset units are both in terms of window height.
//
// Programmer:  Jeremy Meredith
// Creation:    May  5, 2014
//
// Modifications:
// ****************************************************************************
class eavlViewportAnchoredScreenTextAnnotation : public eavlScreenTextAnnotation
{
  protected:
    double vx, vy; // normalized viewport coords
    double dx, dy; // aspect-independent window coordinate offset
  public:
    eavlViewportAnchoredScreenTextAnnotation(eavlWindow *w, const string &txt,
                                             eavlColor c, double s,
                                             double vx, double vy,
                                             double dx, double dy,
                                             double angleDeg = 0.)
        : eavlScreenTextAnnotation(w,txt,c,s,0,0,angleDeg),
          vx(vx), vy(vy), dx(dx), dy(dy)
    {
    }
    virtual void Render(eavlView &view)
    {
        win->SetupForScreenSpace();

        double vl, vr, vb, vt;
        view.GetRealViewport(vl,vr,vb,vt);

        // SetPosition
        x = dx/view.windowaspect + (vl+vr)/2. + vx * (vr-vl)/2.;
        y = dy + (vb+vt)/2. + vy * (vt-vb)/2.;

        eavlScreenTextAnnotation::Render(view);
    }
};

#endif

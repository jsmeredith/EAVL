// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_VIEW_H
#define EAVL_VIEW_H

#include "eavlVector3.h"
#include "eavlPoint3.h"
#include "eavlMatrix4x4.h"

struct eavl3DView
{
    eavlPoint3   from;
    eavlPoint3   at;
    eavlVector3  up;
    float        nearplane;
    float        farplane;

    bool         perspective;
    float        fov; // perspective only
    float        size; // ortho only

    float        zoom;
    float        xpan, ypan;

    //float        xs, ys, zs; ///\todo: would like to add scaling/fullframe
};

struct eavl2DView
{
    float        l,r,t,b;
    //bool logx, logy; ///\todo: would like to add logarithmic scaling

    float xscale; ///< change x scale for non-equal x/y scaling
};

struct eavlView
{
    enum ViewType { EAVL_VIEW_2D, EAVL_VIEW_3D };
    ViewType viewtype;

    // viewport
    float vl, vr, vb, vt;

    eavl3DView view3d;
    eavl2DView view2d;

    eavlMatrix4x4 P, V;

    float w, h; // window width and height
    float windowaspect;
    float viewportaspect;

    float minextents[3];
    float maxextents[3];

    eavlView()
    {
        view3d.perspective = true;
        view2d.xscale = 1;
        vl = -1;  vr = +1;
        vb = -1;  vt = +1;
    }

    void SetupMatrices()
    {
        windowaspect = w / h;

        float l=vl, r=vr, b=vb, t=vt;
        if (viewtype == EAVL_VIEW_2D)
            GetRealViewport(l,r,t,b);

        // the viewport's aspect ratio is in terms of pixels
        viewportaspect = (w*(r-l)) / (h*(t-b));

        // set up projection matrix
        if (viewtype == EAVL_VIEW_2D)
            P.CreateOrthographicProjection(fabs(view2d.t-view2d.b),
                                           +1, -1, viewportaspect);
        else if (view3d.perspective)
            P.CreatePerspectiveProjection(view3d.nearplane, view3d.farplane,
                                          view3d.fov, viewportaspect);
        else
            P.CreateOrthographicProjection(view3d.size,
                                           view3d.nearplane, view3d.farplane,
                                           viewportaspect);

        // set up view matrix
        switch (viewtype)
        {
          case EAVL_VIEW_2D:
            {
                eavlPoint3 at = eavlPoint3((view2d.l+view2d.r)/2.,
                                           (view2d.t+view2d.b)/2.,
                                           0);
                eavlPoint3 from = at + eavlVector3(0,0,1);
                eavlVector3 up = eavlVector3(0,1,0);
                V.CreateView(from,at,up);
                eavlMatrix4x4 M1;
                M1.CreateScale(view2d.xscale, 1, 1);
                V=M1*V;
            }
            break;
          case EAVL_VIEW_3D:
            {
                V.CreateView(view3d.from,view3d.at,view3d.up);
                eavlMatrix4x4 M1, M2;
                M1.CreateTranslate(view3d.xpan,view3d.ypan,0);
                M2.CreateScale(view3d.zoom, view3d.zoom, 1);
                P = M1*M2*P;
            }
            break;
        }
    }

    void GetRealViewport(float &l, float &r, float &b, float &t)
    {
        if (viewtype == EAVL_VIEW_3D)
        {
            // if we don't want to try to clamp the
            // viewport as in 2D, just copy the original
            // viewport as the 'real' one, i.e.:
            l = vl;
            r = vr;
            b = vb;
            t = vt;
            return;
        }

        // We set up a 2D viewport in window coordinates.
        // It has some aspect ratio given the window
        // width and height, but that aspect ratio may
        // not match our view onto the 2D data; we need to clip
        // the original viewport based on the aspect ratio
        // of the window vs the eavl2DView.
        float maxvw = (vr-vl) * w;
        float maxvh = (vt-vb) * h;
        float waspect = maxvw / maxvh;
        float daspect = (view2d.r - view2d.l) / (view2d.t - view2d.b);
        daspect *= view2d.xscale;
        //cerr << "waspect="<<waspect << "   \tdaspect="<<daspect<<endl;
        const bool center = true; // if false, anchor to bottom-left
        if (waspect > daspect)
        {
            float new_w = (vr-vl) * daspect / waspect;
            if (center)
            {
                l = (vl+vr)/2. - new_w/2.;
                r = (vl+vr)/2. + new_w/2.;
            }
            else
            {
                l = vl;
                r = vl + new_w;
            }
            b = vb;
            t = vt;
        }
        else
        {
            float new_h = (vt-vb) * waspect / daspect;
            if (center)
            {
                b = (vb+vt)/2. - new_h/2.;
                t = (vb+vt)/2. + new_h/2.;
            }
            else
            {
                b = vb;
                t = vb + new_h;
            }
            l = vl;
            r = vr;
        }
    }
};

#endif

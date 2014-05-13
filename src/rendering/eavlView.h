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
    double        nearplane;
    double        farplane;

    bool         perspective;
    double        fov; // perspective only
    double        size; // ortho only

    double        zoom;
    double        xpan, ypan;

    //double        xs, ys, zs; ///\todo: would like to add scaling/fullframe
};

struct eavl2DView
{
    double        l,r,t,b;
    bool logx, logy;

    double xscale; ///< change x scale for non-equal x/y scaling
};

struct eavlView
{
    enum ViewType { EAVL_VIEW_2D, EAVL_VIEW_3D };
    ViewType viewtype;

    // viewport
    double vl, vr, vb, vt;

    eavl3DView view3d;
    eavl2DView view2d;

    eavlMatrix4x4 P, V;

    double w, h; // window width and height
    double windowaspect;
    double viewportaspect;

    double minextents[3];
    double maxextents[3];

    eavlView()
    {
        view3d.perspective = true;
        view2d.xscale = 1;
        view2d.logx = false;
        view2d.logy = false;
        vl = -1;  vr = +1;
        vb = -1;  vt = +1;
    }

    void SetupMatrices()
    {
        windowaspect = w / h;

        double l=vl, r=vr, b=vb, t=vt;
        ///\todo: urgent: this is wrong!  should be l,r,b,t!!!
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

    void GetRealViewport(double &l, double &r, double &b, double &t)
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
        double maxvw = (vr-vl) * w;
        double maxvh = (vt-vb) * h;
        double waspect = maxvw / maxvh;
        double daspect = (view2d.r - view2d.l) / (view2d.t - view2d.b);
        daspect *= view2d.xscale;
        //cerr << "waspect="<<waspect << "   \tdaspect="<<daspect<<endl;
        const bool center = true; // if false, anchor to bottom-left
        if (waspect > daspect)
        {
            double new_w = (vr-vl) * daspect / waspect;
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
            double new_h = (vt-vb) * waspect / daspect;
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

    // ------------------------------------------------------------------------

    void Pan3D(double dx, double dy)
    {
        view3d.xpan += dx;
        view3d.ypan += dy;
    }
    void Zoom3D(double zoom)
    {
        double factor = pow(4., zoom);
        view3d.zoom *= factor;
        view3d.xpan *= factor;
        view3d.ypan *= factor;
    }
    void TrackballRotate(double x1, double y1, double x2, double y2)
    {
        eavlMatrix4x4 R1;
        R1.CreateTrackball(-x1,-y1, -x2,-y2);
        eavlMatrix4x4 T1;
        T1.CreateTranslate(-view3d.at);
        eavlMatrix4x4 T2;
        T2.CreateTranslate(view3d.at);
                
        eavlMatrix4x4 V1(V);
        V1.m[0][3]=0;
        V1.m[1][3]=0;
        V1.m[2][3]=0;
        eavlMatrix4x4 V2(V1);
        V2.Transpose();
                
        eavlMatrix4x4 MM = T2 * V2 * R1 * V1 * T1;
                
        view3d.from = MM * view3d.from;
        view3d.at   = MM * view3d.at;
        view3d.up   = MM * view3d.up;
    }

    void Pan2D(double dx, double dy)
    {
        double rvl, rvr, rvt, rvb;
        GetRealViewport(rvl,rvr,rvb,rvt);

        double xpan = dx * (view2d.r - view2d.l) / (rvr - rvl);
        double ypan = dy * (view2d.t - view2d.b) / (rvt - rvb);

        view2d.l -= xpan;
        view2d.r -= xpan;

        view2d.t -= ypan;
        view2d.b -= ypan;
    }
    void Zoom2D(double zoom, bool allowExpand)
    {
        double factor = pow(4., zoom);
        double xc = (view2d.l + view2d.r) / 2.;
        double yc = (view2d.b + view2d.t) / 2.;
        double xs = (view2d.r - view2d.l) / 2.;
        double ys = (view2d.t - view2d.b) / 2.;
        if (allowExpand)
        {
            double rvl, rvr, rvt, rvb;
            GetRealViewport(rvl,rvr,rvb,rvt);

            // If we're zooming in, we first want to expand the
            // viewport if possible before actually having to pull
            // the x/y region in.  We accomplish expanding the
            // viewport the horizontal/vertical direction by
            // (respectively) pulling in the y/x limits while
            // leaving the x/y limits alone.  (Or at least leaving
            // the x/y limits as large as possible.)
            double allowed_x_expansion = (vr - vl) / (rvr - rvl);
            double allowed_y_expansion = (vt - vb) / (rvt - rvb);

            /*
            cerr << "allowx = "<<allowed_x_expansion<<endl;
            cerr << "allowy = "<<allowed_y_expansion<<endl;
            cerr << "factor = "<<factor<<endl;
            cerr << endl;
            */

            if (zoom > 0 && allowed_x_expansion>1.01)
            {
                // not using this:
                //double xfactor = factor;
                //if (allowed_x_expansion > xfactor)
                //    xfactor = 1;
                //else
                //    xfactor /= allowed_x_expansion;

                bool in_l = xc - xs/factor < minextents[0];
                bool in_r = xc + xs/factor > maxextents[0];
                if (in_l && in_r)
                {
                    view2d.l = xc - xs/factor;
                    view2d.r = xc + xs/factor;
                }
                else if (in_l)
                {
                    view2d.l = xc - xs/(factor*factor);
                }
                else if (in_r)
                {
                    view2d.r = xc + xs/(factor*factor);
                }

                view2d.b = yc - ys/factor;
                view2d.t = yc + ys/factor;
            }
            else if (zoom > 0 && allowed_y_expansion>1.01)
            {
                // not using this:
                //double yfactor = factor;
                //if (allowed_y_expansion > yfactor)
                //    yfactor = 1;
                //else
                //    yfactor /= allowed_y_expansion;

                bool in_b = yc - ys/factor < minextents[1];
                bool in_t = yc + ys/factor > maxextents[1];
                if (in_b && in_t)
                {
                    view2d.b = yc - ys/factor;
                    view2d.t = yc + ys/factor;
                }
                else if (in_b)
                {
                    view2d.b = yc - ys/(factor*factor);
                }
                else if (in_t)
                {
                    view2d.t = yc + ys/(factor*factor);
                }

                view2d.l = xc - xs/factor;
                view2d.r = xc + xs/factor;
            }
            else
            {
                view2d.l = xc - xs/factor;
                view2d.r = xc + xs/factor;
                view2d.b = yc - ys/factor;
                view2d.t = yc + ys/factor;
            }
        }
        else // no allowExpand
        {
            view2d.l = xc - xs/factor;
            view2d.r = xc + xs/factor;
            view2d.b = yc - ys/factor;
            view2d.t = yc + ys/factor;
        }
    }

    // ------------------------------------------------------------------------
    //
    // Set up ONLY the viewport for world/screen space
    //
    void SetupViewportForWorld()
    {
        double vl, vr, vt, vb;
        GetRealViewport(vl,vr,vb,vt);
        glViewport(double(w)*(1.+vl)/2.,
                   double(h)*(1.+vb)/2.,
                   double(w)*(vr-vl)/2.,
                   double(h)*(vt-vb)/2.);
    }
    void SetupViewportForScreen()
    {
        glViewport(0, 0, w, h);
    }


    //
    // Set up ONLY the matrices for world/screen space
    //
    void SetupMatricesForWorld()
    {
        glMatrixMode( GL_PROJECTION );
        glLoadMatrixf(P.GetOpenGLMatrix4x4());

        glMatrixMode( GL_MODELVIEW );
        glLoadMatrixf(V.GetOpenGLMatrix4x4());
    }
    void SetupMatricesForScreen()
    {
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glOrtho(-1,1, -1,1, -1,1);

        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();
    }


    //
    // Set up BOTH the matrices and viewport for world/screen space
    //
    void SetupForWorldSpace()
    {
        SetupMatricesForWorld();
        SetupViewportForWorld();
    }
    void SetupForScreenSpace()
    {
        SetupMatricesForScreen();
        SetupViewportForScreen();
    }

};

#endif

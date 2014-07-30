// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_SIMPLE_VR_H
#define EAVL_SCENE_RENDERER_SIMPLE_VR_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"
#include "eavlTexture.h"

// ****************************************************************************
// Class:  eavlSceneRendererSimpleVR
//
// Purpose:
///   A very simple volume renderer.
//
// Programmer:  Jeremy Meredith
// Creation:    July 28, 2014
//
// Modifications:
//
// ****************************************************************************
class eavlSceneRendererSimpleVR : public eavlSceneRenderer
{
    int nsamples;
    vector<float> samples;
    vector<byte> rgba;
    vector<float> depth;
    double mindepth, maxdepth;
  public:
    eavlSceneRendererSimpleVR()
    {
        nsamples = 250;
    }
    virtual ~eavlSceneRendererSimpleVR()
    {
    }

    virtual void StartScene()
    {
        eavlSceneRenderer::StartScene();
        rgba.clear();
        depth.clear();
        rgba.resize(4*view.w*view.h, 0);
        depth.resize(view.w*view.h, 1.0f);

        samples.clear();
        samples.resize(view.w * view.h * nsamples,-1.0f);

        float dist = (view.view3d.from - view.view3d.at).norm();

        eavlPoint3 closest(0,0,-dist+view.size*.5);
        eavlPoint3 farthest(0,0,-dist-view.size*.5);
        mindepth = (view.P * closest).z;
        maxdepth = (view.P * farthest).z;
    }

    virtual void EndScene()
    {
        eavlSceneRenderer::EndScene();

        //
        // composite all samples back-to-front
        // 
        int w = view.w;
        int h = view.h;
#pragma omp parallel for schedule(dynamic,1) collapse(2)
        for (int x=0; x<w; ++x)
        {
            for (int y=0; y<h; ++y)
            {
                eavlColor color(0,0,0,0);
                for (int z=nsamples-1; z>=0; --z)
                {
                    int index3d = (y*view.w + x)*nsamples + z;
                    float value = samples[index3d];
                    if (value<0 || value>1)
                        continue;

                    int colorindex = float(ncolors-1) * value;
                    eavlColor c(colors[colorindex*3+0],
                                colors[colorindex*3+1],
                                colors[colorindex*3+2]);
                    // use a gaussian density function as the opactiy
                    float center = 0.5;
                    float sigma = 0.13;
                    float attenuation = 0.03;
                    float alpha = exp(-(value-center)*(value-center)/(2*sigma*sigma));
                    alpha *= attenuation;
                    color.c[0] = color.c[0] * (1.-alpha) + c.c[0] * alpha;
                    color.c[1] = color.c[1] * (1.-alpha) + c.c[1] * alpha;
                    color.c[2] = color.c[2] * (1.-alpha) + c.c[2] * alpha;
                }

                int index = (y*view.w + x);
                //depth[index] = d;
                rgba[index*4 + 0] = color.c[0]*255.;
                rgba[index*4 + 1] = color.c[1]*255.;
                rgba[index*4 + 2] = color.c[2]*255.;
            }
        }

    }

    virtual bool NeedsGeometryForPlot(int)
    {
        // we're not caching anything; always say we need it
        return true;
    }

    // ------------------------------------------------------------------------

    virtual void StartTetrahedra()
    {
    }

    virtual void EndTetrahedra()
    {
    }

    bool TetBarycentricCoords(eavlPoint3 p0,
                              eavlPoint3 p1,
                              eavlPoint3 p2,
                              eavlPoint3 p3,
                              eavlPoint3 p,
                              float &b0, float &b1, float &b2, float &b3)
    {
        eavlMatrix4x4 Mn(p0.x,p0.y,p0.z, 1,
                         p1.x,p1.y,p1.z, 1,
                         p2.x,p2.y,p2.z, 1,
                         p3.x,p3.y,p3.z, 1);

        eavlMatrix4x4 M0(p.x ,p.y ,p.z , 1,
                         p1.x,p1.y,p1.z, 1,
                         p2.x,p2.y,p2.z, 1,
                         p3.x,p3.y,p3.z, 1);

        eavlMatrix4x4 M1(p0.x,p0.y,p0.z, 1,
                         p.x ,p.y ,p.z , 1,
                         p2.x,p2.y,p2.z, 1,
                         p3.x,p3.y,p3.z, 1);

        eavlMatrix4x4 M2(p0.x,p0.y,p0.z, 1,
                         p1.x,p1.y,p1.z, 1,
                         p.x ,p.y ,p.z , 1,
                         p3.x,p3.y,p3.z, 1);

        eavlMatrix4x4 M3(p0.x,p0.y,p0.z, 1,
                         p1.x,p1.y,p1.z, 1,
                         p2.x,p2.y,p2.z, 1,
                         p.x ,p.y ,p.z , 1);
        float Dn = Mn.Determinant();
        float D0 = M0.Determinant();
        float D1 = M1.Determinant();
        float D2 = M2.Determinant();
        float D3 = M3.Determinant();
        if (Dn<0)
        {
            if (D0>0 || D1>0 || D2>0 || D3>0)
                return false;
        }
        else
        {
            if (D0<0 || D1<0 || D2<0 || D3<0)
                return false;
        }
        b0 = D0/Dn;
        b1 = D1/Dn;
        b2 = D2/Dn;
        b3 = D3/Dn;
        return true;
    }

    virtual void AddTetrahedronVs(double x0, double y0, double z0,
                                  double x1, double y1, double z1,
                                  double x2, double y2, double z2,
                                  double x3, double y3, double z3,
                                  double s0, double s1, double s2, double s3)
    {
        // translate the tet into image space
        eavlPoint3 p[4] = {eavlPoint3(x0,y0,z0),
                           eavlPoint3(x1,y1,z1),
                           eavlPoint3(x2,y2,z2),
                           eavlPoint3(x3,y3,z3)};
        eavlPoint3 s[4];
        eavlPoint3 mine(FLT_MAX,FLT_MAX,FLT_MAX);
        eavlPoint3 maxe(-FLT_MAX,-FLT_MAX,-FLT_MAX);
        for (int i=0; i<4; ++i)
        {
            eavlPoint3 t = view.P * view.V * p[i];
            s[i].x = (t.x*.5+.5) * view.w;
            s[i].y = (t.y*.5+.5) * view.h;
            s[i].z = float(nsamples) * (t.z-mindepth)/(maxdepth-mindepth);
            for (int d=0; d<3; ++d)
            {
                if (s[i][d] < mine[d])
                    mine[d] = s[i][d];
                if (s[i][d] > maxe[d])
                    maxe[d] = s[i][d];
            }
        }

        // discard tets outside the view
        if (maxe[0] < 0)
            return;
        if (maxe[1] < 0)
            return;
        if (maxe[2] < 0)
            return;
        if (mine[0] >= view.w)
            return;
        if (mine[1] >= view.h)
            return;
        if (mine[2] >= nsamples)
            return;

        // clamp extents to what's inside the view
        if (mine[0] < 0)
            mine[0] = 0;
        if (mine[1] < 0)
            mine[1] = 0;
        if (mine[2] < 0)
            mine[2] = 0;
        if (maxe[0] >= view.w)
            maxe[0] = view.w-1;
        if (maxe[1] >= view.h)
            maxe[1] = view.h-1;
        if (maxe[2] >= nsamples)
            maxe[2] = nsamples-1;

        int xmin = mine[0];
        int xmax = ceil(maxe[0]);
        int ymin = mine[1];
        int ymax = ceil(maxe[1]);
        int zmin = mine[2];
        int zmax = ceil(maxe[2]);

        // walk over samples covering the tet in each dimension
        // and sample onto our regular grid
//#pragma omp parallel for schedule(dynamic,1) collapse(2)
        for(int x=xmin; x<xmax; ++x)
        {
            for(int y=ymin; y<ymax; ++y)
            {
                for(int z=zmin; z<zmax; ++z)
                {
                    eavlPoint3 test(x,y,z);
                    float b0,b1,b2,b3;
                    bool isInside = TetBarycentricCoords(s[0],s[1],s[2],s[3],
                                                         test,b0,b1,b2,b3);
                    if (!isInside)
                        continue;
                    float value = b0*s0 + b1*s1 + b2*s2 + b3*s3;
                    int index3d = (y*view.w + x)*nsamples + z;
                    samples[index3d] = value;
                }
            }
        }
    }

    // ------------------------------------------------------------------------

    virtual void AddTriangleVnVs(double, double, double,
                                 double, double, double,
                                 double, double, double,
                                 double, double, double,
                                 double, double, double,
                                 double, double, double,
                                 double, double, double)
    {
    }


    // ------------------------------------------------------------------------

    virtual void AddPointVs(double, double, double, double, double)
    {
    }

    // ------------------------------------------------------------------------

    virtual void AddLineVs(double, double, double,
                           double, double, double,
                           double, double)
    {
    }


    // ------------------------------------------------------------------------
    virtual void Render()
    {
        DrawToScreen();
    }

    void DrawToScreen()
    {
        glColor3f(1,1,1);
        glDisable(GL_BLEND);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);

        // draw the pixel colors
        glDrawPixels(view.w, view.h, GL_RGBA, GL_UNSIGNED_BYTE, &rgba[0]);

        // drawing the Z buffer will overwrite the pixel colors
        // unless you actively prevent it....
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glDepthMask(GL_TRUE);
        // For some bizarre reason, you need GL_DEPTH_TEST enabled for
        // it to write into the Z buffer. 
        glEnable(GL_DEPTH_TEST);

        // draw the z buffer
        glDrawPixels(view.w, view.h, GL_DEPTH_COMPONENT, GL_FLOAT, &depth[0]);

        // set the various masks back to "normal"
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
    }


};


#endif

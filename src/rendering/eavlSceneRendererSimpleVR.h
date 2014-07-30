// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_SIMPLE_VR_H
#define EAVL_SCENE_RENDERER_SIMPLE_VR_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"
#include "eavlTexture.h"
#include "eavlTimer.h"

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
    int th;
    eavlMatrix4x4 XFORM;
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

        eavlMatrix4x4 T,S;
        T.CreateTranslate(1,1,-mindepth);
        S.CreateScale(0.5 * view.w, 0.5*view.h, nsamples/(maxdepth-mindepth));
        XFORM = S * T * view.P * view.V;

        th = eavlTimer::Start();
    }

    virtual void EndScene()
    {
        double frametime = eavlTimer::Stop(th,"vr");
        //cerr << "time per frame = " << frametime << endl;

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
            //cerr << "Dn negative\n";
            if (D0>0 || D1>0 || D2>0 || D3>0)
                return false;
        }
        else
        {
            //cerr << "Dn positive\n";
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
            s[i] = XFORM * p[i];
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

        int xmin = ceil(mine[0]);
        int xmax = floor(maxe[0]);
        int ymin = ceil(mine[1]);
        int ymax = floor(maxe[1]);
        int zmin = ceil(mine[2]);
        int zmax = floor(maxe[2]);

        // ignore tet if it doesn't intersect any sample points
        if (xmin > xmax || ymin > ymax || zmin > zmax)
            return;


        // faster: precalculate partial determinants used for finding barycentric coords

        // From: http://steve.hollasch.net/cgindex/geometry/ptintet.html
        // From: herron@cs.washington.edu (Gary Herron)
        //  Let the tetrahedron have vertices
        //
        //         V0 = (x0, y0, z0)
        //         V1 = (x1, y1, z1)
        //         V2 = (x2, y2, z2)
        //         V3 = (x3, y3, z3)
        //
        // and your test point be
        //
        //         P = (x, y, z).
        //
        // Then the point P is in the tetrahedron if following five determinants all have the same sign.
        //
        //              |x0 y0 z0 1|
        //         Dn = |x1 y1 z1 1|
        //              |x2 y2 z2 1|
        //              |x3 y3 z3 1|
        //
        //              |x  y  z  1|
        //         D0 = |x1 y1 z1 1|
        //              |x2 y2 z2 1|
        //              |x3 y3 z3 1|
        //
        //              |x0 y0 z0 1|
        //         D1 = |x  y  z  1|
        //              |x2 y2 z2 1|
        //              |x3 y3 z3 1|
        //
        //              |x0 y0 z0 1|
        //         D2 = |x1 y1 z1 1|
        //              |x2 y2 z2 1|
        //              |x3 y3 z3 1|
        //
        //              |x0 y0 z0 1|
        //         D3 = |x1 y1 z1 1|
        //              |x3 y3 z3 1|
        //              |x  y  z  1|
        //
        // Some additional notes:
        //     If by chance the D0=0, then your tetrahedron is degenerate (the points are coplanar).
        //     If any other Di=0, then P lies on boundary i (boundary i being that boundary formed by the three points other than Vi).
        //     If the sign of any Di differs from that of D0 then P is outside boundary i.
        //     If the sign of any Di equals that of D0 then P is inside boundary i.
        //     If P is inside all 4 boundaries, then it is inside the tetrahedron.
        //     As a check, it must be that D0 = D1+D2+D3+D4.
        //     The pattern here should be clear; the computations can be extended to simplicies of any dimension. (The 2D and 3D case are the triangle and the tetrahedron).
        //     If it is meaningful to you, the quantities bi = Di/D0 are the usual barycentric coordinates.
        //     Comparing signs of Di and D0 is only a check that P and Vi are on the same side of boundary i. 


        // we genuinely need double precision for some of these calculations, by the way:
        // change these next four to float, and you see obvious artifacts.
        double sx0 = s[0].x, sy0 = s[0].y, sz0 = s[0].z;
        double sx1 = s[1].x, sy1 = s[1].y, sz1 = s[1].z;
        double sx2 = s[2].x, sy2 = s[2].y, sz2 = s[2].z;
        double sx3 = s[3].x, sy3 = s[3].y, sz3 = s[3].z;

        float d_yz_01 = sy0*sz1 - sy1*sz0;
        float d_yz_02 = sy0*sz2 - sy2*sz0;
        float d_yz_03 = sy0*sz3 - sy3*sz0;
        float d_yz_12 = sy1*sz2 - sy2*sz1;
        float d_yz_13 = sy1*sz3 - sy3*sz1;
        float d_yz_23 = sy2*sz3 - sy3*sz2;

        float d_y1_01 = sy0     - sy1   ;
        float d_y1_02 = sy0     - sy2   ;
        float d_y1_03 = sy0     - sy3   ;
        float d_y1_12 = sy1     - sy2   ;
        float d_y1_13 = sy1     - sy3   ;
        float d_y1_23 = sy2     - sy3   ;

        float d_z1_01 = sz0     - sz1   ;
        float d_z1_02 = sz0     - sz2   ;
        float d_z1_03 = sz0     - sz3   ;
        float d_z1_12 = sz1     - sz2   ;
        float d_z1_13 = sz1     - sz3   ;
        float d_z1_23 = sz2     - sz3   ;
 
        float d_yz1_123 = sy1 * d_z1_23 - sy2 * d_z1_13 + sy3 * d_z1_12;
        float d_xz1_123 = sx1 * d_z1_23 - sx2 * d_z1_13 + sx3 * d_z1_12;
        float d_xy1_123 = sx1 * d_y1_23 - sx2 * d_y1_13 + sx3 * d_y1_12;
        float d_xyz_123 = sx1 * d_yz_23 - sx2 * d_yz_13 + sx3 * d_yz_12;

        float d_yz1_023 = sy0 * d_z1_23 - sy2 * d_z1_03 + sy3 * d_z1_02;
        float d_xz1_023 = sx0 * d_z1_23 - sx2 * d_z1_03 + sx3 * d_z1_02;
        float d_xy1_023 = sx0 * d_y1_23 - sx2 * d_y1_03 + sx3 * d_y1_02;
        float d_xyz_023 = sx0 * d_yz_23 - sx2 * d_yz_03 + sx3 * d_yz_02;

        float d_yz1_013 = sy0 * d_z1_13 - sy1 * d_z1_03 + sy3 * d_z1_01;
        float d_xz1_013 = sx0 * d_z1_13 - sx1 * d_z1_03 + sx3 * d_z1_01;
        float d_xy1_013 = sx0 * d_y1_13 - sx1 * d_y1_03 + sx3 * d_y1_01;
        float d_xyz_013 = sx0 * d_yz_13 - sx1 * d_yz_03 + sx3 * d_yz_01;

        float d_yz1_012 = sy0 * d_z1_12 - sy1 * d_z1_02 + sy2 * d_z1_01;
        float d_xz1_012 = sx0 * d_z1_12 - sx1 * d_z1_02 + sx2 * d_z1_01;
        float d_xy1_012 = sx0 * d_y1_12 - sx1 * d_y1_02 + sx2 * d_y1_01;
        float d_xyz_012 = sx0 * d_yz_12 - sx1 * d_yz_02 + sx2 * d_yz_01;

        float Dn =  sx0 * d_yz1_123 - sy0 * d_xz1_123 + sz0 * d_xy1_123 - 1 * d_xyz_123;
        // walk over samples covering the tet in each dimension
        // and sample onto our regular grid
//#pragma omp parallel for schedule(dynamic,1) collapse(2)
        for(int x=xmin; x<=xmax; ++x)
        {
            for(int y=ymin; y<=ymax; ++y)
            {
                for(int z=zmin; z<=zmax; ++z)
                {
                    // raw version: D22 = 2m 1a, d33 = 6m 3a 3m 2a = 9m 5a, d44 = 36m 20a 4m 3a = 40m 23a,
                    //              and five of them = 200m 115a
                    // new version: 12m 18a (d22) 48m 32a (d33) 20m 15a (d44x5) = 80m 65a (2x fewer)
                    //              and with optimization, it's only 16m 12a per sample point (10x fewer)
                    eavlPoint3 test(x,y,z);
                    float value;
                    if (false) // old version
                    {
                        float b0,b1,b2,b3;
                        bool isInside = TetBarycentricCoords(s[0],s[1],s[2],s[3],
                                                             test,b0,b1,b2,b3);
                        if (!isInside)
                            continue;
                        value = b0*s0 + b1*s1 + b2*s2 + b3*s3;
                    }
                    else
                    {
                        float D0 =  x *  d_yz1_123 - y *  d_xz1_123 + z *  d_xy1_123 - 1. * d_xyz_123;
                        float D1 = -x *  d_yz1_023 + y *  d_xz1_023 - z *  d_xy1_023 + 1. * d_xyz_023;
                        float D2 =  x *  d_yz1_013 - y *  d_xz1_013 + z *  d_xy1_013 - 1. * d_xyz_013;
                        float D3 = -x *  d_yz1_012 + y *  d_xz1_012 - z *  d_xy1_012 + 1. * d_xyz_012;
                        if (Dn<0)
                        {
                            // should NEVER fire unless there's a numerical precision error
                            cerr << "Dn negative\n";
                            if (D0>0 || D1>0 || D2>0 || D3>0)
                                continue;
                        }
                        else
                        {
                            //cerr << "Dn positive\n";
                            if (D0<0 || D1<0 || D2<0 || D3<0)
                                continue;
                        }
                        value = (D0*s0 + D1*s1 + D2*s2 + D3*s3) / Dn;
                    }

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

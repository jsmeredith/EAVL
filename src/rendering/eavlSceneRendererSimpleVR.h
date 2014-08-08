// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_SIMPLE_VR_H
#define EAVL_SCENE_RENDERER_SIMPLE_VR_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"
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
    vector<eavlPoint3> p[4];
    vector<float> values;
    double mindepth, maxdepth;
    eavlMatrix4x4 XFORM;
    eavlMatrix4x4 IXFORM;
    eavlView lastview;

    bool PartialDeterminantMode;
  public:
    eavlSceneRendererSimpleVR()
    {
        nsamples = 400;
        PartialDeterminantMode = true;
    }
    virtual ~eavlSceneRendererSimpleVR()
    {
    }

    virtual void StartScene()
    {
        //cerr << "StartScene\n";
        eavlSceneRenderer::StartScene();

        p[0].clear();
        p[1].clear();
        p[2].clear();
        p[3].clear();
        values.clear();
        lastview = eavlView(); // force re-composite
    }

    virtual void EndScene()
    {
        eavlSceneRenderer::EndScene();

    }

    void Composite()
    {
        int th = eavlTimer::Start();
        //cerr << "Composite\n";
        //
        // composite all samples back-to-front
        // 
        //cerr << "color[0] = " <<eavlColor(colors[0],colors[1],colors[2]) << endl;
        float *alphas = new float[ncolors];
        for (int i=0; i<ncolors; ++i)
        {
            float value = float(i)/float(ncolors-1);

            float center = 0.5;
            float sigma = 0.13;
            float alpha = exp(-(value-center)*(value-center)/(2*sigma*sigma));
            //float alpha = .5;

            alphas[i] = alpha;
        }

        int w = view.w;
        int h = view.h;
#pragma omp parallel for collapse(2)
        for (int x=0; x<w; ++x)
        {
            for (int y=0; y<h; ++y)
            {
                eavlColor color(0,0,0,0);
                int minz = nsamples;
                for (int z=nsamples-1; z>=0; --z)
                {
                    int index3d = (y*view.w + x)*nsamples + z;
                    float value = samples[index3d];
                    if (value<0 || value>1)
                        continue;

                    int colorindex = float(ncolors-1) * value;
                    eavlColor c(colors[colorindex*3+0],
                                colors[colorindex*3+1],
                                colors[colorindex*3+2],
                                1.0);
                    // use a gaussian density function as the opactiy
                    float attenuation = 0.02;
                    float alpha = alphas[colorindex];
                    alpha *= attenuation;
                    color.c[0] = color.c[0] * (1.-alpha) + c.c[0] * alpha;
                    color.c[1] = color.c[1] * (1.-alpha) + c.c[1] * alpha;
                    color.c[2] = color.c[2] * (1.-alpha) + c.c[2] * alpha;
                    color.c[3] = color.c[3] * (1.-alpha) + c.c[3] * alpha;
                    minz = z;
                }

                int index = (y*view.w + x);
                if (minz < nsamples)
                {
                    float projdepth = float(minz)*(maxdepth-mindepth)/float(nsamples) + mindepth;
                    depth[index] = .5 * projdepth + .5;
                }
                rgba[index*4 + 0] = color.c[0]*255.;
                rgba[index*4 + 1] = color.c[1]*255.;
                rgba[index*4 + 2] = color.c[2]*255.;
                rgba[index*4 + 3] = color.c[2]*255.;
            }
        }

        delete[] alphas;
        double comptime = eavlTimer::Stop(th,"compositing");

        if (false)
            cerr << "compositing time = "<<comptime << endl;
    }

    // ------------------------------------------------------------------------

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

        if (Dn==0)
        {
            // degenerate tet
            return false;
        }
        else if (Dn<0)
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

    void TetPartialDeterminants(eavlPoint3 s0,
                                eavlPoint3 s1,
                                eavlPoint3 s2,
                                eavlPoint3 s3,
                                float &d_yz1_123,
                                float &d_xz1_123,
                                float &d_xy1_123,
                                float &d_xyz_123,
                                float &d_yz1_023,
                                float &d_xz1_023,
                                float &d_xy1_023,
                                float &d_xyz_023,
                                float &d_yz1_013,
                                float &d_xz1_013,
                                float &d_xy1_013,
                                float &d_xyz_013,
                                float &d_yz1_012,
                                float &d_xz1_012,
                                float &d_xy1_012,
                                float &d_xyz_012,
                                float &Dn)
    {
        double sx0 = s0.x, sy0 = s0.y, sz0 = s0.z;
        double sx1 = s1.x, sy1 = s1.y, sz1 = s1.z;
        double sx2 = s2.x, sy2 = s2.y, sz2 = s2.z;
        double sx3 = s3.x, sy3 = s3.y, sz3 = s3.z;

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
 
        d_yz1_123 = sy1 * d_z1_23 - sy2 * d_z1_13 + sy3 * d_z1_12;
        d_xz1_123 = sx1 * d_z1_23 - sx2 * d_z1_13 + sx3 * d_z1_12;
        d_xy1_123 = sx1 * d_y1_23 - sx2 * d_y1_13 + sx3 * d_y1_12;
        d_xyz_123 = sx1 * d_yz_23 - sx2 * d_yz_13 + sx3 * d_yz_12;

        d_yz1_023 = sy0 * d_z1_23 - sy2 * d_z1_03 + sy3 * d_z1_02;
        d_xz1_023 = sx0 * d_z1_23 - sx2 * d_z1_03 + sx3 * d_z1_02;
        d_xy1_023 = sx0 * d_y1_23 - sx2 * d_y1_03 + sx3 * d_y1_02;
        d_xyz_023 = sx0 * d_yz_23 - sx2 * d_yz_03 + sx3 * d_yz_02;

        d_yz1_013 = sy0 * d_z1_13 - sy1 * d_z1_03 + sy3 * d_z1_01;
        d_xz1_013 = sx0 * d_z1_13 - sx1 * d_z1_03 + sx3 * d_z1_01;
        d_xy1_013 = sx0 * d_y1_13 - sx1 * d_y1_03 + sx3 * d_y1_01;
        d_xyz_013 = sx0 * d_yz_13 - sx1 * d_yz_03 + sx3 * d_yz_01;

        d_yz1_012 = sy0 * d_z1_12 - sy1 * d_z1_02 + sy2 * d_z1_01;
        d_xz1_012 = sx0 * d_z1_12 - sx1 * d_z1_02 + sx2 * d_z1_01;
        d_xy1_012 = sx0 * d_y1_12 - sx1 * d_y1_02 + sx2 * d_y1_01;
        d_xyz_012 = sx0 * d_yz_12 - sx1 * d_yz_02 + sx2 * d_yz_01;

        Dn =  sx0 * d_yz1_123 - sy0 * d_xz1_123 + sz0 * d_xy1_123 - 1 * d_xyz_123;
    }

    virtual void AddTetrahedronVs(double x0, double y0, double z0,
                                  double x1, double y1, double z1,
                                  double x2, double y2, double z2,
                                  double x3, double y3, double z3,
                                  double s0, double s1, double s2, double s3)
    {
        p[0].push_back(eavlPoint3(x0,y0,z0));
        p[1].push_back(eavlPoint3(x1,y1,z1));
        p[2].push_back(eavlPoint3(x2,y2,z2));
        p[3].push_back(eavlPoint3(x3,y3,z3));
        values.push_back(s0);
        values.push_back(s1);
        values.push_back(s2);
        values.push_back(s3);
    }

    void ChangeView()
    {
        //cerr << "ChangeView\n";
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
        IXFORM = XFORM;
        IXFORM.Invert();
    }


    void Sample()
    {
        int samples_tried = 0;
        int samples_eval = 0;
        int tets_eval = 0;
        double zdepth_sum = 0;
        //cerr << "Sample\n";
        int th = eavlTimer::Start();
        int n = p[0].size();

#pragma omp parallel for schedule(dynamic,1)
        for (int tet = 0; tet < n ; tet++)
        {
            // translate the tet into image space
            eavlPoint3 s[4];
            eavlPoint3 mine(FLT_MAX,FLT_MAX,FLT_MAX);
            eavlPoint3 maxe(-FLT_MAX,-FLT_MAX,-FLT_MAX);
            for (int i=0; i<4; ++i)
            {
                s[i] = XFORM * p[i][tet];
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
                continue;
            if (maxe[1] < 0)
                continue;
            if (maxe[2] < 0)
                continue;
            if (mine[0] >= view.w)
                continue;
            if (mine[1] >= view.h)
                continue;
            if (mine[2] >= nsamples)
                continue;

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
                continue;

            tets_eval++;

            // we genuinely need double precision for some of these calculations, by the way:
            // change these next four to float, and you see obvious artifacts.

            float d_yz1_123=0, d_xz1_123=0, d_xy1_123=0, d_xyz_123=0;
            float d_yz1_023=0, d_xz1_023=0, d_xy1_023=0, d_xyz_023=0;
            float d_yz1_013=0, d_xz1_013=0, d_xy1_013=0, d_xyz_013=0;
            float d_yz1_012=0, d_xz1_012=0, d_xy1_012=0, d_xyz_012=0;
            float Dn=1, iDn=1;
            if (PartialDeterminantMode)
            {
                TetPartialDeterminants(s[0],s[1],s[2],s[3],
                                   d_yz1_123, d_xz1_123, d_xy1_123, d_xyz_123,
                                   d_yz1_023, d_xz1_023, d_xy1_023, d_xyz_023,
                                   d_yz1_013, d_xz1_013, d_xy1_013, d_xyz_013,
                                   d_yz1_012, d_xz1_012, d_xy1_012, d_xyz_012,
                                   Dn);
                if (Dn == 0)
                {
                    // degenerate
                    continue;
                }
                iDn = 1. / Dn;
            }

            zdepth_sum += 1+zmax-zmin;

            // in theory, we know whether or not CLAMP_Z_EXTENTS
            // is useful for every tetrahedron based on the 
            // z depth of this tet's bounding box.  I think
            // it has to be 2 or more to be helpful.  we can
            // make this a per-tet decision
#define CLAMP_Z_EXTENTS
#ifdef CLAMP_Z_EXTENTS
            if (d_xy1_123==0 ||
                d_xy1_023==0 ||
                d_xy1_013==0 ||
                d_xy1_012==0)
            {
                // degenerate tetrahedron
                continue;
            }

            float i123 = 1. / d_xy1_123;
            float i023 = 1. / d_xy1_023;
            float i013 = 1. / d_xy1_013;
            float i012 = 1. / d_xy1_012;
#endif

            // also, don't necessarily need to pull the samples
            // from memory here; might be better to do them
            // later and assume they're cached if necessary
            float s0 = values[tet*4+0];
            float s1 = values[tet*4+1];
            float s2 = values[tet*4+2];
            float s3 = values[tet*4+3];

            // walk over samples covering the tet in each dimension
            // and sample onto our regular grid
            //#pragma omp parallel for schedule(dynamic,1) collapse(2)
            for(int x=xmin; x<=xmax; ++x)
            {
                for(int y=ymin; y<=ymax; ++y)
                {
                    int startindex = (y*view.w + x)*nsamples;

                    float t0 =  x *  d_yz1_123 - y *  d_xz1_123 - 1. * d_xyz_123;
                    float t1 = -x *  d_yz1_023 + y *  d_xz1_023 + 1. * d_xyz_023;
                    float t2 =  x *  d_yz1_013 - y *  d_xz1_013 - 1. * d_xyz_013;
                    float t3 = -x *  d_yz1_012 + y *  d_xz1_012 + 1. * d_xyz_012;

                    // timing note:
                    // without updating Z extents and just using bounding box,
                    // we accepted only about 10-15% of samples.  (makes sense,
                    // given the size of a tet within a bounding cube)
                    // noise.silo, 400 samples, sample time = .080 to 0.087 with clamping
                    //                                      = .083 to 0.105 without clamping
                    // without omp, max 1.0 (no clamp) drops to max 0.75 (clamp)
                    // in other words, CLAMP_Z_EXTENTS is a factor of 20-25% faster on noise, best case
                    // but on rect_cube, it's a factor of 270% faster (2.7x) on rect_cube!
                    // on noise_256, it's a small slowdown, 7%.  (i think we're doing more divisions)
                    // maxes sense; once we're about 1 sample per tet, the extra divisions we need to do
                    // are only used about once, so it's better to just try out the samples
#ifdef CLAMP_Z_EXTENTS
                    float newzmin = zmin;
                    float newzmax = zmax;
                    
                    float z0 = -t0 * i123;
                    float z1 = +t1 * i023;
                    float z2 = -t2 * i013;
                    float z3 = +t3 * i012;

                    if (-i123 < 0) { newzmin = std::max(newzmin,z0); } else { newzmax = std::min(newzmax,z0); }
                    if (+i023 < 0) { newzmin = std::max(newzmin,z1); } else { newzmax = std::min(newzmax,z1); }
                    if (-i013 < 0) { newzmin = std::max(newzmin,z2); } else { newzmax = std::min(newzmax,z2); }
                    if (+i012 < 0) { newzmin = std::max(newzmin,z3); } else { newzmax = std::min(newzmax,z3); }
                    newzmin = ceil(newzmin);
                    newzmax = floor(newzmax);
                    for(int z=newzmin; z<=newzmax; ++z)
#else
                    for(int z=zmin; z<=zmax; ++z)
#endif
                    {
                        samples_tried++;
                        float value;
                        if (!PartialDeterminantMode)
                        {
                            // Mode where we calculate the full barycentric
                            // coordinates from scratch each time.
                            float b0,b1,b2,b3;
                            bool isInside =
                                TetBarycentricCoords(s[0],s[1],s[2],s[3],
                                                     eavlPoint3(x,y,z),b0,b1,b2,b3);
                            if (!isInside)
                                continue;
                            value = b0*s0 + b1*s1 + b2*s2 + b3*s3;
                        }
                        else
                        {
                            // Mode where we pre-calculate partial determinants
                            // to avoid a bunch of redundant arithmetic.
                            float D0 = t0 + z *  d_xy1_123;
                            float D1 = t1 - z *  d_xy1_023;
                            float D2 = t2 + z *  d_xy1_013;
                            float D3 = t3 - z *  d_xy1_012;

                            // explicit calculation, without precalculating the constant and x/y terms
                            //float D0 =  x *  d_yz1_123 - y *  d_xz1_123 + z *  d_xy1_123 - 1. * d_xyz_123;
                            //float D1 = -x *  d_yz1_023 + y *  d_xz1_023 - z *  d_xy1_023 + 1. * d_xyz_023;
                            //float D2 =  x *  d_yz1_013 - y *  d_xz1_013 + z *  d_xy1_013 - 1. * d_xyz_013;
                            //float D3 = -x *  d_yz1_012 + y *  d_xz1_012 - z *  d_xy1_012 + 1. * d_xyz_012;
#ifndef CLAMP_Z_EXTENTS
                            // if we already clamped the Z extents, we know every sample
                            // is already inside the tetrahedron!
                            if (Dn<0)
                            {
                                // should NEVER fire unless there's a numerical precision error
                                //cerr << "Dn negative\n";
                                if (D0>0 || D1>0 || D2>0 || D3>0)
                                    continue;
                            }
                            else
                            {
                                //cerr << "Dn positive\n";
                                if (D0<0 || D1<0 || D2<0 || D3<0)
                                    continue;
                            }
#endif                            
                            value = (D0*s0 + D1*s1 + D2*s2 + D3*s3) * iDn;
                        }

                        int index3d = startindex + z;
                        samples[index3d] = value;
                        samples_eval++;
                    }
                }
            }
        }
        double sampletime = eavlTimer::Stop(th,"sample");

        if (false)
        {
            // NOTE: These values should be ignored if OpenMP was enabled above:
            cerr << zdepth_sum/double(n) << " average z samples per tet\n";
            cerr << samples_eval << " out of " << samples_tried << " ("
                 << (100.*double(samples_eval)/double(samples_tried)) << "%) samples\n";
            cerr << tets_eval << " out of " << n << " ("
                 << (100.*double(tets_eval)/double(n)) << "%) tetrahedra\n";

            cerr << "w="<<view.w<<" h="<<view.h << endl;
            cerr << "Sample time = "<<sampletime << endl;
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
        if (lastview != view)
        {
            ChangeView();
            Sample();
            lastview = view;
        }
        Composite();
    }

    virtual unsigned char *GetRGBAPixels()
    {
        return &rgba[0];
    }

    virtual float *GetDepthPixels()
    {    
        return &depth[0];
    }
};


#endif
